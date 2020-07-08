import numpy as np
np.random.seed(42)
import math

from keras import metrics
from keras.callbacks import Callback
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

from . import Hex10Model
from .data_reader import load_dataset


def get_index_dict(input_data):
    """
    Create index - word/label dict from list input
    @params : List of lists
    @returns : Index - Word/label dictionary
    """
    result = dict()
    vocab = set()
    i = 1
    # Flatten list and get indices
    for element in [word for sentence in input_data for word in sentence]:
        if element not in vocab:
            result[i]=element
            i+=1
            vocab.add(element)
    return result

class Hex10BiLSTM(Hex10Model):
    def __init__(self, params):
        super(Hex10BiLSTM, self).__init__(params)

    def train_and_predict(self):
        hidden_units = self.params["hidden_units"]
        dropout_drop_prob = self.params["dropout"]
        batch_size = self.params["batch_size"]
        model_path = self.params["model_path"]

        # Load data: [[token11, token12, ...],[token21,token22,...]]
        # and label: [[label11, label12, ...],[label21,label22,...]]
        X_train_data, y_train_data = load_dataset("train.dat", seq2seq=False)
        X_dev_data, y_dev_data = load_dataset("dev.dat", seq2seq=False)
        X_test_data, y_test_data = load_dataset("test.dat", seq2seq=False)

        # Get index -> label and label -> index dictionaries
        index_to_char = get_index_dict(X_train_data + X_dev_data + X_test_data)
        char_to_index = dict((v, k) for k, v in index_to_char.items())
        input_vocab_size = len(char_to_index)

        # one-hot embedding matrix with zero-vector for padding
        embedding_matrix = np.vstack((np.zeros(input_vocab_size), np.eye(input_vocab_size)))

        index_to_label = get_index_dict(y_train_data + y_dev_data + y_test_data)
        label_to_index = dict((v, k) for k, v in index_to_label.items())

        # Get indexed data and labels
        X_train_index = [[char_to_index[char] for char in word] for word in X_train_data]
        X_dev_index = [[char_to_index[char] for char in word] for word in X_dev_data]
        X_test_index = [[char_to_index[char] for char in word] for word in X_test_data]

        y_train_index = [[label_to_index[label] for label in word] for word in y_train_data]
        y_dev_index = [[label_to_index[label] for label in word] for word in y_dev_data]
        y_test_index = [[label_to_index[label] for label in word] for word in y_test_data]

        # For batch training:
        # Pad additional 0 elements at the end for the last batch:
        X_train_padded = X_train_index + [[0] for _ in range(
            math.ceil(len(X_train_index) / batch_size) * batch_size - len(X_train_index))]
        X_dev_padded = X_dev_index + [[0] for _ in
                                      range(math.ceil(len(X_dev_index) / batch_size) * batch_size - len(X_dev_index))]
        X_test_padded = X_test_index + [[0] for _ in range(
            math.ceil(len(X_test_index) / batch_size) * batch_size - len(X_test_index))]

        y_train_padded = y_train_index + [[0] for _ in range(
            math.ceil(len(y_train_index) / batch_size) * batch_size - len(y_train_index))]
        y_dev_padded = y_dev_index + [[0] for _ in
                                      range(math.ceil(len(y_dev_index) / batch_size) * batch_size - len(y_dev_index))]

        # Get maximum sentence length to pad instances
        max_word_length = max(map(lambda x: len(x), X_train_data + X_dev_data))

        # Get the number of classes:
        output_vocab_size = len(index_to_label.items())

        # Pad for all inputs
        X_train = sequence.pad_sequences(X_train_padded, maxlen=max_word_length)
        X_dev = sequence.pad_sequences(X_dev_padded, maxlen=max_word_length, padding='post')
        X_test = sequence.pad_sequences(X_test_padded, maxlen=max_word_length, padding='post')

        # For categorical cross_entropy we need matrices representing the classes:
        # Note that we pad after doing the transformation into the matrix!
        y_train = sequence.pad_sequences(
            np.asarray([np_utils.to_categorical(y_label, output_vocab_size + 1) for y_label in y_train_padded]),
            maxlen=max_word_length)
        y_dev = sequence.pad_sequences(
            np.asarray([np_utils.to_categorical(y_label, output_vocab_size + 1) for y_label in y_dev_padded]),
            maxlen=max_word_length, padding='post')
        # We do not need to pad y_test, since we can just use y_test_index

        model = Sequential()
        model.add(Embedding(input_vocab_size + 1,
                            input_vocab_size,
                            input_length=max_word_length,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=False,
                            batch_input_shape=(batch_size, max_word_length)))
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
        model.add(Dropout(dropout_drop_prob))
        model.add(TimeDistributed(Dense(output_vocab_size + 1, activation='softmax')))
        model.compile('adagrad', 'categorical_crossentropy', metrics=[metrics.categorical_accuracy])


        class WordAccuracyModelCheckpointer(Callback):
            def __init__(self):
                super(WordAccuracyModelCheckpointer).__init__()

            def on_train_begin(self, logs={}):
                self.best_acc = -1

            def word_accuracy(self, pred, truth, truncate_pred=True):
                """Computes word-based accuracy
                """
                tp = 0
                for word_pred, word_truth in zip(pred, truth):
                    word_pred = [index_to_label[char_pred.tolist().index(max(char_pred))] for char_pred in word_pred]
                    if truncate_pred:
                        word_pred = word_pred[:len(word_truth)]
                    word_truth = [index_to_label[char_truth] for char_truth in word_truth]
                    if word_pred == word_truth:
                        tp += 1
                return tp / len(truth)

            def on_epoch_end(self, batch, logs={}):
                predict = np.asarray(model.predict(self.validation_data[0], batch_size=batch_size))
                self.accs = self.word_accuracy(predict, y_dev_index)
                if self.accs > self.best_acc:
                    self.best_acc = self.accs
                    model.save(model_path)
                    print("New best word accuracy, saving model...")

                print(" Word accuracy: %f" % (self.accs))
                return

        checkpointer = WordAccuracyModelCheckpointer()
        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=self.params["epochs"],
                  validation_data=(X_dev, y_dev),
                  callbacks=[checkpointer],
                  shuffle=False)

        model.load_weights(self.params["model_path"])

        # Get class probabilities for the test set:
        predictions = model.predict(X_test, batch_size=batch_size)
        predictions_indices = np.argmax(predictions, axis=2)

        # Use unpadded inputs and omit padding
        result = []
        for letters_input, letters_truth, letters_pred in zip(X_test_index, y_test_index, predictions_indices):
            word_input = "".join([index_to_char[i] for i in letters_input])
            word_truth = "".join([index_to_label[i] for i in letters_truth])
            word_pred = "".join([index_to_label[i] for i in letters_pred[:len(letters_truth)]])

            # remove alignment helpers
            word_truth = word_truth.replace("_MYJOIN_", "").replace("EMPTY", "")
            word_pred = word_pred.replace("_MYJOIN_", "").replace("EMPTY", "")

            result.append((word_input, word_pred, word_truth))
        return result