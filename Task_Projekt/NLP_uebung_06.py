import numpy as np
import nltk
from tensorflow import keras


# Aufgabe 2.1
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

        return first_sentences, second_sentences


def score_writer(file: str, scores: [float]):
    """
        Writes the given list of scores to the file with the given filename.
        :param file: [str], path of local data set (sentences)
        :param scores: [list], scores of labeled sentences
    """
    with open(file, mode='w', encoding='utf8') as file:
        file.write('\n'.join(str(score) for score in scores))


# Aufgabe 2.2
def load_vectors(size=40000, filename="./wiki-news-300d-1M.vec"):
    """
    read them into a Python dictionary that maps every token to the corresponding vector,
    load only the first 20000 or 40000 lines of the file
    :param size: [int], size of loading vector (default value: 40000)
    :param filename: [str], file's name (default value: "./wiki-news-300d-1M.vec")
    :return: data [dict], word vector reference
    """
    with open(filename, 'r', encoding='utf-8', newline='\n') as file:
        # initial
        data = {}
        num = 0

        next(file)  # skip the first line
        # read from second line
        for line in file.readlines():
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array([float(i) for i in tokens[1:]]).reshape((1, 300))

            # count iteration
            num += 1
            if num >= size:
                break

        return data


def aver_sentence(data: dict, sentence: str):
    """
    implement a function that embeds each sentence as the average of the embeddings of its tokens

    :param data: [dict], wiki-news-300d-1M dataset
    :param sentence: [str]
    :return: the average of the embeddings of its tokens: value_aver
    """
    def map_token(token, data):
        for key in data:
            if key == token:
                return np.array(data[key])
        return np.zeros((1, 300))

    tokens = nltk.word_tokenize(sentence)
    value = np.zeros((1, 300), dtype=float)
    for token in tokens:
        value += map_token(token, data)
    value_aver = value / len(tokens)
    return value_aver


def get_sentences_vector(data: dict, sentences: list):
    """
    return a sentences vector with word vector average
    :param data: [dict], wiki-news-300d-1M dataset
    :param sentences: [list], list of first sentence
    :return sentences_vectors [array], word vector average of each sentence
    """
    # initial
    sentences_vectors = np.zeros((1, 300), dtype=float)

    for sentence in sentences:
        sentences_vectors = np.vstack((sentences_vectors, aver_sentence(data, sentence)))
    sentences_vectors = np.delete(sentences_vectors, 0, axis=0)

    return sentences_vectors


# Aufgabe 2.3
def modeling():
    # define the model layer
    # define multi-input layers
    first_sentence_input = keras.layers.Input(shape=(300, ))
    second_sentence_input = keras.layers.Input(shape=(300, ))
    # concatenate input layer
    concatenation = keras.layers.Concatenate(axis=1)([first_sentence_input, second_sentence_input])
    # define dropout layer 1
    dropout_1 = keras.layers.Dropout(0.3)(concatenation)
    # define dense layer 1
    dense_layer_1 = keras.layers.Dense(300, activation=keras.activations.relu)(dropout_1)
    # define dropout layer 2
    dropout_2 = keras.layers.Dropout(0.3)(dense_layer_1)
    # define dense layer 2
    output = keras.layers.Dense(1, activation=keras.activations.sigmoid)(dropout_2)

    # build model
    model = keras.Model([first_sentence_input, second_sentence_input], output)

    # print model structure
    print(model.summary())
    print("Defined the model.")

    # compile the model
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error,
                  metrics=[keras.metrics.mean_squared_error])
    print("Compiled the model.")

    return model


if __name__ == '__main__':
    # Aufgabe 2.1
    dev_scores, dev_first_sentences, dev_second_sentences = read_label_dataset(file="development-dataset.txt")
    train_scores, train_first_sentences, train_second_sentences = read_label_dataset(file="training-dataset.txt")
    test_scores, test_first_sentences, test_second_sentences = read_label_dataset(file="test-hex06-dataset.txt")

    # Aufgabe 2.2
    data = load_vectors("./wiki-news-300d-1M.vec")
    dev_first_sentences_vec = get_sentences_vector(data, dev_first_sentences)
    dev_second_sentences_vec = get_sentences_vector(data, dev_second_sentences)

    train_first_sentences_vec = get_sentences_vector(data, train_first_sentences)
    train_second_sentences_vec = get_sentences_vector(data, train_second_sentences)

    test_first_sentences_vec = get_sentences_vector(data, test_first_sentences)
    test_second_sentences_vec = get_sentences_vector(data, test_second_sentences)

    # Aufgabe 2.3
    model = modeling()
    # train the model and observe the mean squared error on the development set
    model.fit([train_first_sentences_vec, train_second_sentences_vec], train_scores,
              validation_data=([dev_first_sentences_vec, dev_second_sentences_vec], dev_scores),
              batch_size=100, epochs=300)
    print("Trained the model.")

    # evaluate the model on the test set
    res = model.evaluate([test_first_sentences_vec, test_second_sentences_vec], test_scores)
    print(res)


