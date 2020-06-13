import numpy as np
import os
import math
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras import metrics
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, Bidirectional, TimeDistributed
from sklearn.metrics import f1_score
from keras.models import load_model
from util import utils

np.random.seed(42)


def run(params):
    batch_size = params["batch_size"]
    model_path = params["model_path"]
    predict_file = params["predict_file"]

    # Load data: [[token11, token12, ...],[token21,token22,...]]
    # and label: [[label11, label12, ...],[label21,label22,...]]
    X_train_data, y_train_data = utils.get_task_data("pos", utils.load_dataset("ner_eng_bio.train"))
    X_dev_data, y_dev_data = utils.get_task_data("pos", utils.load_dataset("ner_eng_bio.dev"))
    X_test_data, y_test_data = utils.get_task_data("pos", utils.load_dataset("ner_eng_bio.test"))

    # Load pretrained embeddings and get reverse index
    mat_embedding, word_to_index = utils.get_pretrained_embeddings(params["embeddings"])
    index_to_word = dict((v, k) for k, v in word_to_index.items())

    # Get index -> label and label -> index dictionaries
    index_to_label = utils.get_index_dict(y_train_data + y_dev_data + y_test_data)
    label_to_index = dict((v, k) for k, v in index_to_label.items())

    # Get indexed data and labels - 1 is the OOV index
    X_train_index = [[word_to_index.get(word.lower(), 1) for word in sentence] for sentence in X_train_data]
    X_dev_index = [[word_to_index.get(word.lower(), 1) for word in sentence] for sentence in X_dev_data]
    X_test_index = [[word_to_index.get(word.lower(), 1) for word in sentence] for sentence in X_test_data]

    y_train_index = [[label_to_index[label] for label in sentence] for sentence in y_train_data]
    y_dev_index = [[label_to_index[label] for label in sentence] for sentence in y_dev_data]
    y_test_index = [[label_to_index[label] for label in sentence] for sentence in y_test_data]


    # For batch training:
    # Pad additional 0 elements at the end for the last batch:
    X_train_padded = X_train_index + [[0] for _ in range(math.ceil(len(X_train_index)/batch_size)*batch_size-len(X_train_index))]
    X_dev_padded = X_dev_index + [[0] for _ in range(math.ceil(len(X_dev_index)/batch_size)*batch_size-len(X_dev_index))]
    X_test_padded = X_test_index + [[0] for _ in range(math.ceil(len(X_test_index)/batch_size)*batch_size-len(X_test_index))]

    y_train_padded = y_train_index + [[0] for _ in range(math.ceil(len(y_train_index)/batch_size)*batch_size-len(y_train_index))]
    y_dev_padded = y_dev_index + [[0] for _ in range(math.ceil(len(y_dev_index)/batch_size)*batch_size-len(y_dev_index))]

    # Get maximum sentence length to pad instances
    max_sentence_length = max(map(lambda x: len(x), X_train_data + X_dev_data))

    # Get the number of classes:
    number_of_classes = len(index_to_label.items())

    # Pad for all inputs
    X_train = sequence.pad_sequences(X_train_padded, maxlen=max_sentence_length)
    X_dev = sequence.pad_sequences(X_dev_padded, maxlen=max_sentence_length, padding='post')
    X_test = sequence.pad_sequences(X_test_padded, maxlen=max_sentence_length, padding='post')

    # For categorical cross_entropy we need matrices representing the classes:
    # Note that we pad after doing the transformation into the matrix!
    y_train = sequence.pad_sequences(np.asarray([np_utils.to_categorical(y_label,number_of_classes+1) for y_label in y_train_padded]), maxlen=max_sentence_length)
    y_dev = sequence.pad_sequences(np.asarray([np_utils.to_categorical(y_label,number_of_classes+1) for y_label in y_dev_padded]), maxlen=max_sentence_length, padding='post')
    # We do not need to pad y_test, since we can just use y_test_index


    # ------------------------------------------------
    #             2.2 Bi-LSTM Model
    # ------------------------------------------------


    def build_model(embedding_matrix, max_sentence_length, number_of_classes):
        model = Sequential()
        model.add(Embedding(len(embedding_matrix),
                            len(embedding_matrix[0]),
                            input_length=max_sentence_length,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=False,
                            batch_input_shape=(batch_size, max_sentence_length)))

        ####################################
        #                                  #
        model.add(Dropout(params["dropout"]))
        model.add(Bidirectional(LSTM(params["hidden_units"], return_sequences=True)))
        model.add(Dropout(params["dropout"]))
        model.add(TimeDistributed(Dense(number_of_classes + 1, activation='softmax')))
        #                                  #
        ####################################

        model.compile('adagrad', 'categorical_crossentropy', metrics=[metrics.categorical_accuracy])
        return model

    # ------------------------------------------------
    #            2.4 F1 Model Checkpointer
    # ------------------------------------------------

    class F1ScoreModelCheckpointer(Callback):
        def __init__(self, model):
            super(F1ScoreModelCheckpointer).__init__()
            self.model = model

        def on_train_begin(self, logs={}):
            self.best_f1 = 0.0

        def on_epoch_end(self, batch, logs={}):

            ####################################
            #                                  #
            #   add your implementation here   #
            #                                  #
            # 1. Get predictions               #
            predict = self.model.predict(self.validation_data[0], batch_size=batch_size)
            # self.validation_data[0] --> X_dev
            # self.validation_data[1] --> Y_dev

            # 2. Flatten all outputs and       #
            #    remove padding                #
            dev_pred = []
            dev_truth = []
            for sent_pred, sent_truth in zip(predict, y_dev_index):
                # to remove padding: use 2nd arg--> y_dev_index
                # if want to with padding: use 2nd arg--> y_dev_padded
                for word_pred, lab in zip(sent_pred, sent_truth):
                    dev_pred.append(index_to_label[word_pred.tolist().index(max(word_pred))])
                    dev_truth.append(index_to_label[lab])
                    if word_pred.tolist().index(max(word_pred)) == 0:
                        print("Warning, PADDING label got predicted!")
            # 3. Compute f1 score              #
            self.f1s = f1_score(dev_pred, dev_truth, labels=list(index_to_label.values()), average='macro',
                                zero_division=0)
            # 4. Store best model              #
            if self.f1s > self.best_f1:
                self.best_f1 = self.f1s
                self.model.save(model_path)
                print(f"get better f1 score, saving model to {model_path}")
            #                                  #
            ####################################

            print(" F1 score %f" % (self.f1s))
            return


    def train(model):
        if params["checkpointer"] == "f1":
            checkpointer = F1ScoreModelCheckpointer(model)
        elif params["checkpointer"] == "acc":
            checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True,
                                       monitor='val_categorical_accuracy', mode='max')
        else:
            raise ValueError('Unknown checkpointer "{}"'.format(params["checkpointer"]))

        model.fit(X_train,
                  y_train,
                  batch_size=batch_size,
                  epochs=params["epochs"],
                  validation_data=(X_dev,y_dev),
                  callbacks=[checkpointer],
                  shuffle=False)

    def predict(model):
        model.load_weights(model_path)

        # Get class probabilities for the test set:
        predictions = model.predict(X_test, batch_size=batch_size)

        # Compute F1 score for test set:
        test_pred = []
        test_truth = []
        for sent_pred, sent_truth in zip(predictions, y_test_index):
            for word_pred, lab in zip(sent_pred, sent_truth):
                test_pred.append(index_to_label[word_pred.tolist().index(max(word_pred))])
                test_truth.append(index_to_label[lab])
                if word_pred.tolist().index(max(word_pred)) == 0:
                    print("Warning, PADDING label got predicted!")
        # task 2.2
        f1_test = f1_score(test_truth, test_pred, list(index_to_label.values()), average='macro',
                           zero_division=0)
        print("F1 score on test set: {}".format(f1_test))

        # Use unpadded inputs and omit padding
        # Output in conll format: line_number(int)  word(str)   predict_label(str)  true_label(str)
        prediction_result = utils.get_prediction_results(X_test_index, y_test_index, predictions, index_to_word, index_to_label)

        with open(predict_file, 'w') as file:
            for document in prediction_result:
                for line in document:
                    file.write("\t".join(line) + "\n")

    model = build_model(mat_embedding, max_sentence_length, number_of_classes)
    train(model)
    predict(model)


if __name__ == '__main__':
    model_path = os.path.join("results", "lstm.model")
    predict_file = os.path.join("results", "predictions.txt")

    # # task 2.2
    # params = {"model_path": model_path,
    #           "predict_file": predict_file,
    #           "checkpointer": "acc",        # "acc" or "f1"
    #           "batch_size": 10,
    #           "dropout": 0.5,
    #           "hidden_units": 100,
    #           "epochs": 10,
    #           "embeddings": "glove.6B.50d.txt"}
    # run(params)

    # task 2.4
    # model = load_model(model_path)
    params = {"model_path": model_path,
              "predict_file": predict_file,
              "checkpointer": "f1",  # "acc" or "f1"
              "batch_size": 10,
              "dropout": 0.5,
              "hidden_units": 100,
              "epochs": 10,
              "embeddings": "glove.6B.50d.txt"}
    run(params)
    print("---------------")
    # task 2.5
    params = {"model_path": model_path,
              "predict_file": predict_file,
              "checkpointer": "f1",  # "acc" or "f1"
              "batch_size": 20,
              "dropout": 0.5,
              "hidden_units": 100,
              "epochs": 20,
              "embeddings": "glove.6B.50d.txt"}
    run(params)
    # print("---------------")
    # params = {"model_path": model_path,
    #           "predict_file": predict_file,
    #           "checkpointer": "f1",  # "acc" or "f1"
    #           "batch_size": 40,
    #           "dropout": 0.2,
    #           "hidden_units": 100,
    #           "epochs": 40,
    #           "embeddings": "glove.6B.50d.txt"}
    # run(params)
    # print("---------------")
    # params = {"model_path": model_path,
    #           "predict_file": predict_file,
    #           "checkpointer": "f1",  # "acc" or "f1"
    #           "batch_size": 50,
    #           "dropout": 0.3,
    #           "hidden_units": 100,
    #           "epochs": 100,
    #           "embeddings": "glove.6B.50d.txt"}
    # run(params)
    # print("---------------")