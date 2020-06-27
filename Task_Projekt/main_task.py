import numpy as np
import nltk
import random
from tensorflow import keras
import os
from preprocessing import load_data as ld
from preprocessing import convert_vector as cv
from processing import modeling
from postprocessing import record


def loading():
    """
    use node code to load the sentences
    convert sentences to word vector
    """
    # load data sets
    dev_scores, dev_first_sentences, dev_second_sentences = ld.read_label_dataset(file="development-dataset.txt")
    train_scores, train_first_sentences, train_second_sentences = ld.read_label_dataset(file="training-dataset.txt")
    test_scores, test_first_sentences, test_second_sentences = ld.read_label_dataset(file="test-hex06-dataset.txt")
    val_first_sentences, val_second_sentences = ld.read_unlabel_dataset(file="test-scoreboard-dataset.txt")

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

    return (train_first_sentences_vec, train_second_sentences_vec, train_scores,
            dev_first_sentences_vec, dev_second_sentences_vec, dev_scores,
            test_first_sentences_vec, test_second_sentences_vec, test_scores,
            val_first_sentences_vec, val_second_sentences_vec)


def opimazation(model, search_size):
    """
    use random search to find the best hyper parameter
    param: model, [estimator], the model of estimator
    return params, [list], list of searched hyper parameters
    """
    random.seed(233)
    for i in range(search_size):
        # set random layer
        layers_num = random.randint(1, 10)
        # set number of Filters, Filters size(kernel_size) and Stride sizes in each layer
        layers = [(random.randint(200, 400), random.randint(100, 200), random.randint(1, 4)) for _ in range(layers_num)]


if __name__ == '__main__':
    # gpu/cpu transform
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load data sets
    (train_first_sentences_vec, train_second_sentences_vec, train_scores,
     dev_first_sentences_vec, dev_second_sentences_vec, dev_scores,
     test_first_sentences_vec, test_second_sentences_vec, test_scores,
     val_first_sentences_vec, val_second_sentences_vec) = loading()

    # assign hyper parameter
    hyperparameter = {"Dropout_rate": (0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
                      "hidden_layer_size": (300, 150, 75, 30, 10),
                      "activation": ('relu', 'relu', 'relu', 'relu', 'relu','sigmoid')}

    # build MLP model
    MLP = modeling.modeling(hyperparameter)

    # define callback function
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=40, verbose=0, mode='auto',
                                         baseline=None, restore_best_weights=True)
    save = keras.callbacks.ModelCheckpoint(filepath='result/task_MLP.hdf5', monitor='val_loss', mode='auto',
                                           save_best_only=True, save_weights_only=False, verbose=1)

    # train the model and observe the mean squared error on the development set
    MLP.fit([train_first_sentences_vec, train_second_sentences_vec], train_scores,
            validation_data=([dev_first_sentences_vec, dev_second_sentences_vec], dev_scores),
            batch_size=30, epochs=300, verbose=0,
            callbacks=[stop, save])

    print("Trained the model.")

    # evaluate the model on the test set
    res = MLP.evaluate([test_first_sentences_vec, test_second_sentences_vec], test_scores)

    # predict the test data set of scoreboard
    pre = MLP.predict([val_first_sentences_vec, val_second_sentences_vec])

    # save the prediction of unlabeled data
    record.score_writer('result/test_result.txt', pre.reshape(-1).tolist())
