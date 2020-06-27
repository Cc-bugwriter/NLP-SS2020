import numpy as np
import nltk
from tensorflow import keras
from preprocessing import load_data as ld
from preprocessing import convert_vector as cv
from processing import modeling

if __name__ == '__main__':
    # load data sets
    dev_scores, dev_first_sentences, dev_second_sentences = ld.read_label_dataset(file="development-dataset.txt")
    train_scores, train_first_sentences, train_second_sentences = ld.read_label_dataset(file="training-dataset.txt")
    test_scores, test_first_sentences, test_second_sentences = ld.read_label_dataset(file="test-hex06-dataset.txt")

    # convert word to vector
    data = cv.load_vectors(size=40000, filename="./wiki-news-300d-1M.vec")
    dev_first_sentences_vec = cv.get_sentences_vector(data, dev_first_sentences)
    dev_second_sentences_vec = cv.get_sentences_vector(data, dev_second_sentences)

    train_first_sentences_vec = cv.get_sentences_vector(data, train_first_sentences)
    train_second_sentences_vec = cv.get_sentences_vector(data, train_second_sentences)

    test_first_sentences_vec = cv.get_sentences_vector(data, test_first_sentences)
    test_second_sentences_vec = cv.get_sentences_vector(data, test_second_sentences)

    # assign hyper parameter
    hyperparameter = {"Dropout_rate": (0.3, 0.3, 0.3),
                      "hidden_layer_size": (300, 150),
                      "activation": ('relu', 'relu', 'sigmoid')}

    # build MLP model
    MLP = modeling.modeling(hyperparameter)

    # train the model and observe the mean squared error on the development set
    MLP.fit([train_first_sentences_vec, train_second_sentences_vec], train_scores,
              validation_data=([dev_first_sentences_vec, dev_second_sentences_vec], dev_scores),
              batch_size=1, epochs=300, verbose=1)
    print("Trained the model.")

