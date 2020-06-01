import numpy as np
import io
import nltk


# Aufgabe 2.1
def read_dataset(file="test-hex06-dataset.txt", labeled=True):
    """
        read txt data set, adapt for labeled and unlabeled datasets
        :param file: [str], path of local data set (default value： "test-hex06-dataset.txt")
        :param labeled: [boolean], whether the data set is labeled or not
        :return:
        scores: [list], list of scores when labeled
        first_sentences: [list], list of first sentence
        second_sentences: [list], list of second sentence
    """
    # assign path of data sets
    path = "./Data/"
    with open(f"{path}{file}", 'r', encoding='utf-8') as file:
        if labeled:
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
        else:
            # unlabeled data
            first_sentences = []
            second_sentences = []
            for line in file.readlines():
                first_sentence, second_sentence = line.strip().split('\t')
                first_sentences.append(first_sentence)
                second_sentences.append(second_sentence)

            return first_sentences, second_sentences


# Aufgabe 2.2
def load_vectors(filename="./wiki-news-300d-1M.vec"):
    """
    read them into a Python dictionary that maps every token to the corresponding vector,
    load only the first 20000 or 40000 lines of the file
    :param filename: [str], file's name (default value: "./wiki-news-300d-1M.vec")
    :return: data [dict]
    """
    fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    num = 0
    limit = 40000
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(i) for i in tokens[1:]]).reshape((1, 300))

        num += 1
        if num > limit:
            break
    return data


def aver_sentence(data, sentence):
    """
    implement a function that embeds each sentence as the average of the embeddings of its tokens

    :param data: wiki-news-300d-1M dataset
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


if __name__ == '__main__':
    # nltk.download('punkt')
    scores, first_sentences, second_sentences = read_dataset(file="test-hex06-dataset.txt", labeled=True)

    # Aufgabe 2.2
    data = load_vectors("./wiki-news-300d-1M.vec")
    first_sentences_vectors = []
    second_sentences_vectors = []

    for first_sentence in first_sentences:
        first_sentences_vectors.append(aver_sentence(data, first_sentence))
    for second_sentence in second_sentences:
        first_sentences_vectors.append(aver_sentence(data, second_sentence))

    # convert to array
    first_sentences_vectors = np.array(first_sentences_vectors)
    second_sentences_vectors = np.array(second_sentences_vectors)

