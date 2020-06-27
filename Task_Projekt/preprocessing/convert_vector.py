import numpy as np
import nltk


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