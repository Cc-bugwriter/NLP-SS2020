
import os
import math
import time
import random

from keras.preprocessing import sequence
import tensorflow as tf
import keras

import numpy as np
import nltk

def read_label_dataset(file="test-hex06-dataset.txt"):
    """
        read txt data set, adapt for labeled data sets
        :param file: [str], path of local data set (default value： "test-hex06-dataset.txt")
        :param labeled: [boolean], whether the data set is labeled or not
        :return:
        scores: [list], list of scores when labeled
        first_sentences: [list], list of first sentence
        second_sentences: [list], list of second sentence
    """
    # assign path of data sets
    path = "./DATA/"
    with open(f"{path}{file}", 'r', encoding='utf-8') as file:
        # labeled data
        scores = []
        first_sentences = []
        second_sentences = []
        for line in file.readlines():
            score, first_sentence, second_sentence = line.strip().split('\t')
            scores.append(float(score))
            first_sentences.append(first_sentence)
            second_sentences.append(second_sentence)
        # string to list
        first_sentences = [sentence.split() for sentence in first_sentences]
        second_sentences = [sentence.split() for sentence in second_sentences]

        return scores, first_sentences, second_sentences

def read_unlabel_dataset(file: str):
    """
        read txt data set, adapt for unlabeled data sets
        :param file: [str], path of local data set
        :return:
        first_sentences: [list], list of first sentence
        second_sentences: [list], list of second sentence
    """
    # assign path of data sets
    path = "./DATA/"
    with open(f"{path}{file}", 'r', encoding='utf-8') as file:
        # unlabeled data
        first_sentences = []
        second_sentences = []
        for line in file.readlines():
            first_sentence, second_sentence = line.strip().split('\t')
            first_sentences.append(first_sentence)
            second_sentences.append(second_sentence)
        # string to list
        first_sentences = [sentence.split() for sentence in first_sentences]
        second_sentences = [sentence.split() for sentence in second_sentences]

        return first_sentences, second_sentences


# load data sets
dev_scores, dev_first_sentences, dev_second_sentences = read_label_dataset(file="development-dataset.txt")
train_scores, train_first_sentences, train_second_sentences = read_label_dataset(file="training-dataset.txt")
test_scores, test_first_sentences, test_second_sentences = read_label_dataset(file="test-hex06-dataset.txt")
val_first_sentences, val_second_sentences = read_unlabel_dataset(file="test-scoreboard-dataset.txt")


def get_pretrained_embeddings(size=40000, filename="./wiki-news-300d-1M.vec"):
    """
    xxxxx
    """
    embedding_dimension = 300
    with open(filename, 'r', encoding='utf-8', newline='\n') as file:

        # reset the file pointer to the beginning of the file
        file.seek(0)
        num = 0
        # +=2 because we reserve one entry in the embedding matrix for OOV tokens and another one for our padding token

        embedding_matrix = np.empty((size+2, embedding_dimension), dtype=np.float32)
        word_to_embedding = {}

        embedding_matrix[0] = np.zeros(embedding_dimension)  # index 0 for __PADDING__
        embedding_matrix[1] = np.random.randn(embedding_dimension)  # index 1 for OOV words
        word_to_embedding["__PADDING__"] = 0
        word_to_embedding["__OOV__"] = 1

        next(file)
        # starting with index 2, we enter the regular words
        for i, line in enumerate(file, 2):
            parts = line.split()
            word_to_embedding[parts[0]] = i
            embedding_matrix[i] = np.array(parts[1:], dtype=np.float32)

            # count iteration
            num += 1
            if num >= size:
                break
    return embedding_matrix, word_to_embedding

embedding_matrix, word_to_embedding = get_pretrained_embeddings(size=80000, filename="./wiki-news-300d-1M.vec")


max_sentence_length = max(map(lambda x: len(x), train_first_sentences + train_second_sentences
                              + dev_first_sentences + dev_second_sentences
                              + test_first_sentences + test_second_sentences))

def pad_sentence(sentences, word_to_index, max_sentence_length):

    data_index = [[word_to_index.get(word, 1) for word in sentence] for sentence in sentences]
    padded_data = sequence.pad_sequences(data_index, maxlen=max_sentence_length)

    return padded_data

# Pad for all inputs
train_first_sentences_data = pad_sentence(train_first_sentences, word_to_embedding, max_sentence_length)
train_second_sentences_data = pad_sentence(train_second_sentences, word_to_embedding, max_sentence_length)

dev_first_sentences_data = pad_sentence(dev_first_sentences, word_to_embedding, max_sentence_length)
dev_second_sentences_data = pad_sentence(dev_second_sentences, word_to_embedding, max_sentence_length)

test_first_sentences_data = pad_sentence(test_first_sentences, word_to_embedding, max_sentence_length)
test_second_sentences_data = pad_sentence(test_second_sentences, word_to_embedding, max_sentence_length)

val_first_sentences_data = pad_sentence(val_first_sentences, word_to_embedding, max_sentence_length)
val_second_sentences_data = pad_sentence(val_second_sentences, word_to_embedding, max_sentence_length)

def get_model(hyperparameter: dict):
    embedding_layer = tf.keras.layers.Embedding(len(embedding_matrix),
                                len(embedding_matrix[0]),
                                weights=[embedding_matrix],
                                input_length=max_sentence_length,
                                trainable=False)


    lstm_layer = tf.keras.layers.LSTM(hyperparameter["lstm_hidden_units"])


    sequence_1_input = tf.keras.layers.Input(shape=(max_sentence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    y1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = tf.keras.layers.Input(shape=(max_sentence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y2 = lstm_layer(embedded_sequences_2)

    merged = tf.keras.layers.Concatenate(axis=1)([y1, y2])
    merged = tf.keras.layers.Dropout(hyperparameter["Dense_Dropout_rate"][0])(merged)
    merged = tf.keras.layers.BatchNormalization()(merged)

    # append each hidden layer
    for i in range(len(hyperparameter["hidden_layer_size"])):
        # define dense layer i
        merged = tf.keras.layers.Dense(hyperparameter["hidden_layer_size"][i],
                                                activation=hyperparameter["activation"][i])(merged)

        # define dropout layer i
        merged = tf.keras.layers.Dropout(hyperparameter["Dense_Dropout_rate"][i])(merged)
        merged = tf.keras.layers.BatchNormalization()(merged)

    output = tf.keras.layers.Dense(1, activation=hyperparameter["activation"][-1])(merged)

    model = tf.keras.Model(inputs=[sequence_1_input, sequence_2_input], outputs=output)

    model.compile(optimizer='adam', loss=keras.losses.mean_squared_logarithmic_error,
                  metrics=[keras.metrics.mean_squared_error])
    model.summary()
    return model


def opimazation(model, search_size):
    """
    use random search to find the best hyper parameter
    param: model, [str], the model of estimator
    return params, [list], list of searched hyper parameters
    """
    # assign activation function candidate
    candidate_space = ['relu', 'selu', 'elu', 'hard_sigmoid', 'sigmoid']
    random.seed(613)

    # initial output dict
    params_dict = {"hyperparameter": [],
                   "score": []
                   }
    # timer start
    time_start = time.time()

    # random search
    for i in range(search_size):
        # set random layer
        layers_num = random.randint(1, 3)
        # set drop rate, number of hidden neurons and activation function in each layer
        hyperparameter = {"Dense_Dropout_rate": [(0.1*random.randint(2, 5)) for _ in range(layers_num+1)],
                          "hidden_layer_size": [(random.randint(int(2**(7-j)), 2**(8-j)))
                                                for j in range(layers_num)],
                          "lstm_hidden_units": 300,
                          "activation": [(candidate_space[random.randint(0, 4)]) for _ in range(layers_num+1)]}

        LSTM = get_model(hyperparameter)

        print(f'searching...({i+1}/{search_size})')

        LSTM.fit([train_first_sentences_data, train_second_sentences_data], train_scores,
                 validation_data=([dev_first_sentences_data, dev_second_sentences_data], dev_scores),
                 batch_size=30, epochs=30, verbose=1,
                 callbacks=[stop])

        # evaluate the model on the test set
        res = LSTM.evaluate([test_first_sentences_data, test_second_sentences_data], test_scores)

        params_dict["hyperparameter"].append(hyperparameter)
        params_dict["score"].append(res[1])

    # timer end
    time_end = time.time()
    # print searching time
    print('searching time cost', time_end - time_start, 's')

    # find the best hyper parameter in search ranking
    index = np.flatnonzero(params_dict["score"] == sorted(params_dict["score"])[0])
    best_param = params_dict["hyperparameter"][int(index)]

    return best_param, params_dict

# stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=40, verbose=1, mode='auto',
#                                          baseline=None, restore_best_weights=True)
#
# save = tf.keras.callbacks.ModelCheckpoint(filepath='result/task_LSTM.hdf5', monitor='val_loss', mode='auto',
#                                            save_best_only=True, save_weights_only=False, verbose=1)
# # random search
# best_param, params = opimazation(None, 10)
#
# # build LSTM model
#
# # hyperparameter = {"Dense_Dropout_rate": (0.3, 0.3),
# #                   "hidden_layer_size": (300, 150),
# #                   "lstm_hidden_units": 300,
# #                   "activation": ('relu', 'relu', 'sigmoid')}
#
# LSTM = get_model(best_param)
#
# # train the model and observe the mean squared error on the development set
#
# LSTM.fit([train_first_sentences_data, train_second_sentences_data], train_scores,
#          validation_data=([dev_first_sentences_data, dev_second_sentences_data], dev_scores),
#          batch_size=30, epochs=50, verbose=1,
#          callbacks=[stop, save])
#
# print("Trained the model.")
#
# # evaluate the model on the test set
# res = LSTM.evaluate([test_first_sentences_data, test_second_sentences_data], test_scores)
#
# # predict the test data set of scoreboard
# pre = LSTM.predict([val_first_sentences_data, val_second_sentences_data])




def score_writer(file: str, scores: [float]):
    """
        Writes the given list of scores to the file with the given filename.
        :param file: [str], path of local data set (sentences)
        :param scores: [list], scores of labeled sentences
    """
    with open(file, mode='w', encoding='utf8') as file:
        file.write('\n'.join(str(score) for score in scores))

# save the prediction of unlabeled data
score_writer('result/lstm_test_result.txt', pre.reshape(-1).tolist())
















