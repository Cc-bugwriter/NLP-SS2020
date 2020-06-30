import numpy as np
import nltk
import random
from tensorflow import keras
import os
from preprocessing import load_data as ld
from preprocessing import convert_vector as cv
from preprocessing import preprose_attackes_txt as pa
from processing import modeling
from postprocessing import record
import time
import pickle


def loading(fast=True):
    """
    use node code to load the sentences
    convert sentences to word vector
    """
    if not fast:
        # load data sets
        dev_scores, dev_first_sentences, dev_second_sentences = ld.read_label_dataset(file="development-dataset.txt")
        train_scores, train_first_sentences, train_second_sentences = ld.read_label_dataset(file="training-dataset.txt")
        test_scores, test_first_sentences, test_second_sentences = ld.read_label_dataset(file="test-hex06-dataset.txt")
        val_first_sentences, val_second_sentences = ld.read_unlabel_dataset(file="test-scoreboard-dataset.txt")

        # recover the visual attack with inverse VIPER
        test_first_sentences = pa.get_unattacked_sentences(test_first_sentences)
        test_second_sentences = pa.get_unattacked_sentences(test_second_sentences)
        val_first_sentences = pa.get_unattacked_sentences(val_first_sentences)
        val_second_sentences = pa.get_unattacked_sentences(val_second_sentences)

        # assign embedding_matrix
        _, word_to_embedding = cv.get_pretrained_embeddings()

        # get max_sentence_length
        max_sentence_length = max(map(lambda x: len(x), train_first_sentences + train_second_sentences
                                      + dev_first_sentences + dev_second_sentences
                                      + test_first_sentences + test_second_sentences))

        if model == "MLP":
            # convert word to vector
            data = cv.load_vectors(size=80000, filename="./wiki-news-300d-1M.vec")
            dev_first_sentences_vec = cv.get_sentences_vector(data, dev_first_sentences)
            dev_second_sentences_vec = cv.get_sentences_vector(data, dev_second_sentences)

            train_first_sentences_vec = cv.get_sentences_vector(data, train_first_sentences)
            train_second_sentences_vec = cv.get_sentences_vector(data, train_second_sentences)

            test_first_sentences_vec = cv.get_sentences_vector(data, test_first_sentences)
            test_second_sentences_vec = cv.get_sentences_vector(data, test_second_sentences)

            val_first_sentences_vec = cv.get_sentences_vector(data, val_first_sentences)
            val_second_sentences_vec = cv.get_sentences_vector(data, val_second_sentences)

        elif model == "LSTM":
            # convert word to vector
            dev_first_sentences_vec = cv.pad_sentence(dev_first_sentences, word_to_embedding, max_sentence_length)
            dev_second_sentences_vec = cv.pad_sentence(dev_second_sentences, word_to_embedding, max_sentence_length)

            train_first_sentences_vec = cv.pad_sentence(train_first_sentences, word_to_embedding, max_sentence_length)
            train_second_sentences_vec = cv.pad_sentence(train_second_sentences, word_to_embedding, max_sentence_length)

            test_first_sentences_vec = cv.pad_sentence(test_first_sentences, word_to_embedding, max_sentence_length)
            test_second_sentences_vec = cv.pad_sentence(test_second_sentences, word_to_embedding, max_sentence_length)

            val_first_sentences_vec = cv.pad_sentence(val_first_sentences, word_to_embedding, max_sentence_length)
            val_second_sentences_vec = cv.pad_sentence(val_second_sentences, word_to_embedding, max_sentence_length)

        # save to local (prepare for fast load)
        np.save(f'DATA/fast_load/{model}/dev_first_sentences_vec.npy', dev_first_sentences_vec)
        np.save(f'DATA/fast_load/{model}/dev_second_sentences_vec.npy', dev_second_sentences_vec)
        np.save(f'DATA/fast_load/{model}/dev_scores.npy', dev_scores)
        np.save(f'DATA/fast_load/{model}/train_first_sentences_vec.npy', train_first_sentences_vec)
        np.save(f'DATA/fast_load/{model}/train_second_sentences_vec.npy', train_second_sentences_vec)
        np.save(f'DATA/fast_load/{model}/train_scores.npy', train_scores)
        np.save(f'DATA/fast_load/{model}/test_first_sentences_vec.npy', test_first_sentences_vec)
        np.save(f'DATA/fast_load/{model}/test_second_sentences_vec.npy', test_second_sentences_vec)
        np.save(f'DATA/fast_load/{model}/test_scores.npy', test_scores)
        np.save(f'DATA/fast_load/{model}/val_first_sentences_vec.npy', val_first_sentences_vec)
        np.save(f'DATA/fast_load/{model}/val_second_sentences_vec.npy', val_second_sentences_vec)

        return (train_first_sentences_vec, train_second_sentences_vec, train_scores,
                dev_first_sentences_vec, dev_second_sentences_vec, dev_scores,
                test_first_sentences_vec, test_second_sentences_vec, test_scores,
                val_first_sentences_vec, val_second_sentences_vec)

    else:
        train_first_sentences_vec = np.load(f"DATA/fast_load/{model}/train_first_sentences_vec.npy")
        train_second_sentences_vec = np.load(f"DATA/fast_load/{model}/train_second_sentences_vec.npy")
        train_scores = np.load(f"DATA/fast_load/{model}/train_scores.npy")
        dev_first_sentences_vec = np.load(f"DATA/fast_load/{model}/dev_first_sentences_vec.npy")
        dev_second_sentences_vec = np.load(f"DATA/fast_load/{model}/dev_second_sentences_vec.npy")
        dev_scores = np.load(f"DATA/fast_load/{model}/dev_scores.npy")
        test_first_sentences_vec = np.load(f"DATA/fast_load/{model}/test_first_sentences_vec.npy")
        test_second_sentences_vec = np.load(f"DATA/fast_load/{model}/test_second_sentences_vec.npy")
        test_scores = np.load(f"DATA/fast_load/{model}/test_scores.npy")
        val_first_sentences_vec = np.load(f"DATA/fast_load/{model}/val_first_sentences_vec.npy")
        val_second_sentences_vec = np.load(f"DATA/fast_load/{model}/val_second_sentences_vec.npy")

        return (train_first_sentences_vec, train_second_sentences_vec, train_scores,
                dev_first_sentences_vec, dev_second_sentences_vec, dev_scores,
                test_first_sentences_vec, test_second_sentences_vec, test_scores,
                val_first_sentences_vec, val_second_sentences_vec)


def opimazation(model, search_size, max_deep):
    """
    use random search to find the best hyper parameter
    param: model, [str], the model of estimator
    param: search_size, [int], xxx
    param: max_deep, [int], xxx
    return params, [list], list of searched hyper parameters
    """
    # assign activation function candidate
    candidate_space = ['relu', 'selu', 'elu', 'hard_sigmoid', 'sigmoid']
    random.seed(23333)

    # initial output dict
    params_dict = {"hyperparameter": [],
                   "score": []
                   }
    # timer start
    time_start = time.time()

    # random search
    for i in range(search_size):
        # set random layer
        layers_num = random.randint(1, max_deep)
        # set drop rate, number of hidden neurons and activation function in each layer
        hyperparameter = {"Dropout_rate": [(0.1*random.randint(1, 5)) for _ in range(layers_num+1)],
                          "hidden_layer_size": [(random.randint(int(2**(8-j)), 2**(9-j)))
                                                for j in range(layers_num)],
                          "activation": [(candidate_space[random.randint(0, 4)]) for _ in range(layers_num+1)]}

        if model == "MLP":
            Preceptron = modeling.modeling_MLP(hyperparameter)
        elif model == "LSTM":
            Preceptron = modeling.modeling_LSTM_MLP(hyperparameter)

        print(f'searching...({i+1}/{search_size})')

        Preceptron.fit([train_first_sentences_vec, train_second_sentences_vec], train_scores,
                validation_data=([dev_first_sentences_vec, dev_second_sentences_vec], dev_scores),
                batch_size=30, epochs=6, verbose=1,
                callbacks=[stop])

        # evaluate the model on the test set
        res = Preceptron.evaluate([test_first_sentences_vec, test_second_sentences_vec], test_scores)

        # release memory
        keras.backend.clear_session()

        params_dict["hyperparameter"].append(hyperparameter)
        params_dict["score"].append(res[1])

    # timer end
    time_end = time.time()
    # print searching time
    print('searching time cost', time_end - time_start, 's')

    # find the best hyper parameter in search ranking
    index = np.flatnonzero(params_dict["score"] == sorted(params_dict["score"])[0])
    best_param = params_dict["hyperparameter"][int(index)]

    # save searched paramters
    params_save = open(f'param_{model}.pkl', 'wb')
    pickle.dump(params_dict, params_save)
    params_save.clo

    return best_param, params_dict


if __name__ == '__main__':
    # params_space (controller)
    fast = True
    model = "MLP"
    search_size = 50
    max_deep = 8
    train = False

    # gpu/cpu transform
    if model == "MLP":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load data sets
    (train_first_sentences_vec, train_second_sentences_vec, train_scores,
     dev_first_sentences_vec, dev_second_sentences_vec, dev_scores,
     test_first_sentences_vec, test_second_sentences_vec, test_scores,
     val_first_sentences_vec, val_second_sentences_vec) = loading(fast=fast)

    if train:
        # define callback function
        stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=0, mode='auto',
                                             baseline=None, restore_best_weights=True)
        save = keras.callbacks.ModelCheckpoint(filepath=f'result/task_{model}.hdf5', monitor='val_loss', mode='auto',
                                               save_best_only=True, save_weights_only=False, verbose=1)

        # random search
        best_param, params = opimazation(model, search_size, max_deep)

        # build Preceptron model
        if model == "MLP":
            Preceptron = modeling.modeling_MLP(best_param)
        elif model == "LSTM":
            Preceptron = modeling.modeling_LSTM_MLP(best_param)

        # merge all available data set
        train_first_sentences_vec = np.vstack(
            (train_first_sentences_vec, dev_first_sentences_vec, test_first_sentences_vec,
             train_second_sentences_vec, dev_second_sentences_vec, test_second_sentences_vec))
        train_second_sentences_vec = np.vstack(
            (train_second_sentences_vec, dev_second_sentences_vec, test_second_sentences_vec,
             train_first_sentences_vec, dev_first_sentences_vec, test_first_sentences_vec))
        train_scores = np.hstack((train_scores, dev_scores, test_scores,
                                  train_scores, dev_scores, test_scores))

        # train the model and observe the mean squared error on the development set
        Preceptron.fit([train_first_sentences_vec, train_second_sentences_vec], train_scores,
                       validation_data=([dev_first_sentences_vec, dev_second_sentences_vec], dev_scores),
                       batch_size=30, epochs=100, verbose=0,
                       callbacks=[stop, save])

        print("Trained the model.")
    else:
        # load trained Model
        Preceptron = keras.models.load_model(f'./result/task_{model}.hdf5')

    # evaluate the model on the test set
    res = Preceptron.evaluate([test_first_sentences_vec, test_second_sentences_vec], test_scores)

    # predict the test data set of scoreboard
    pre_part_1 = Preceptron.predict([val_first_sentences_vec, val_second_sentences_vec])
    pre_part_2 = Preceptron.predict([val_second_sentences_vec, val_first_sentences_vec])

    pre = np.mean(pre_part_1, pre_part_2)

    # save the prediction of unlabeled data
    record.score_writer('result/scores.txt', pre.reshape(-1).tolist())
