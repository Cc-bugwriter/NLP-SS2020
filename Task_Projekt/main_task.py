import numpy as np
import nltk
from tensorflow import keras
from preprocessing import load_data as ld

if __name__ == '__main__':
    # Aufgabe 2.1
    dev_scores, dev_first_sentences, dev_second_sentences = ld.read_label_dataset(file="development-dataset.txt")
    train_scores, train_first_sentences, train_second_sentences = ld.read_label_dataset(file="training-dataset.txt")
    test_scores, test_first_sentences, test_second_sentences = ld.read_label_dataset(file="test-hex06-dataset.txt")


