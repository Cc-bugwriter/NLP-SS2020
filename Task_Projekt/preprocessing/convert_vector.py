import numpy as np
import nltk
from tensorflow import keras


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


def aver_sentence(data: dict, tokens):
    """
    implement a function that embeds each sentence as the average of the embeddings of its tokens

    :param data: [dict], wiki-news-300d-1M dataset
    :param tokens: [list]
    :return: the average of the embeddings of its tokens: value_aver
    """
    def map_token(token, data):
        for key in data:
            if key == token:
                return np.array(data[key])
        return np.random.rand(1, 300)

    # tokens = nltk.word_tokenize(sentence)
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


def get_pretrained_embeddings(size=80000, filename="./wiki-news-300d-1M.vec"):
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


def pad_sentence(sentences, word_to_embedding, max_sentence_length=56):
    """
    xxxx
    """
    data_index = [[word_to_embedding.get(word, 1) for word in sentence] for sentence in sentences]
    padded_data = keras.preprocessing.sequence.pad_sequences(data_index, maxlen=max_sentence_length)

    return padded_data
