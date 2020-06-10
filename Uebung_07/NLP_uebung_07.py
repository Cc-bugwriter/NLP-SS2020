import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Aufgabe 3.1
def read_data_sets(reviews_path: str, labels_path: str):
    """
    load review file and label file
    :param reviews_path [str], reviews data path
    :param labels_path [str], labels data path
    :return reviews [list], reviews content
    :return labels_onehot [array], label embeddings
    """
    # initial list
    reviews = []
    labels = []

    # read files
    with open(reviews_path, 'r') as reviews_file, open(labels_path, 'r') as labels_file:
        for review, label in zip(reviews_file, labels_file):
            # synchronous load
            if not review or not label:
                continue
            reviews.append(review.strip('. \n'))
            labels.append(label.strip('\n'))

        # modify label encoder
        labels_onehot = np.zeros((len(labels), 2), dtype=np.int32)
        for i, label in enumerate(labels):
            if label == 'POS':
                labels_onehot[i, 0] = 1
            else:
                labels_onehot[i, 1] = 1

    return reviews, labels_onehot


def load_word2vec_embeddings(embedding_path: str):
    """
    loads the word2vec embeddings trained in 2.2
    :param embedding_path [str], word2vec embedding path
    :return embedding_dict [dict], dictionary of embedding
    :return embedding_size [list], size of embedding
    """
    # initial dict
    embedding_dict = {}

    # read file
    with open(embedding_path, 'r', encoding='utf-8', newline='\n') as file:
        # read first line, which contains size of embedding
        embedding_size = next(file).strip().split()
        # convert str element to int
        embedding_size = list(map(int, embedding_size))

        for line in file.readlines():
            embedding = line.strip().split()
            embedding_dict[embedding[0]] = np.array([float(i) for i in embedding[1:]]).reshape((1, embedding_size[1]))

    return embedding_dict, embedding_size


def comput_power_mean(reviews: list, embedding_dict: dict, embedding_size: list):
    """
    compute power mean of reviews sentence
    :param reviews [list], reviews content
    :param embedding_dict [dict], dictionary of embedding
    :param embedding_size [list], size of embedding
    :return reviews_vec [array], power mean of reviews sentence
    """
    # assign out-of-vocabulary embedding
    oov_embedding = np.zeros(embedding_size[1]).reshape(1, -1)

    # initial representation
    reviews_vec = np.zeros((len(reviews), embedding_size[1]*4))

    # ergodicity sentence
    for i, review in enumerate(reviews):
        # initial intermediate variables of each review sentence
        sentence_vec = np.zeros((len(review.strip().split(' ')), embedding_size[1]))

        # memory the mapping of each word in sentence
        for j, word in enumerate(review.strip().split(' ')):
            sentence_vec[j] = embedding_dict.get(word, oov_embedding)

        # power mean along each word
        arithmetic_average = np.mean(sentence_vec, axis=0)
        minimum = np.min(sentence_vec, axis=0)
        maximum = np.min(sentence_vec, axis=0)
        quadratic_mean = np.square(sentence_vec).mean(axis=0)

        # concatenate all power mean
        reviews_vec[i] = np.concatenate((arithmetic_average, minimum, maximum, quadratic_mean), axis=0)

    return reviews_vec


def MLP_modell(reviews_train, label_train, reviews_test, labels_test, reviews_dev, label_dev):
    """
    MLP which takes an embedding of a review as an input, applies softmax
    and uses cross-entropy as the loss function
    :param reviews_train [array], power mean of reviews sentence (training set)
    :param label_train [array], onehot labels (training set)
    :param reviews_test [array], power mean of reviews sentence (test set)
    :param labels_test [array], onehot labels  (test set)
    :param reviews_dev [array], power mean of reviews sentence (development set)
    :param label_dev [array], onehot labels (development set)
    :return model [class], MLP Regressor of Keras
    :return history [callback], record of fitting process
    """
    # build model
    model = Sequential()
    model.add(Dense(300, activation='softmax', input_shape=(reviews_train.shape[1],)))
    model.add(Dense(50, activation='softmax'))
    model.add(Dense(label_train.shape[1], activation='softmax'))

    # print model structure
    print(model.summary())
    print("Defined the model.")

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print("Compiled the model.")

    # define callback functions
    my_callbacks = [
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=10, verbose=0, mode='auto', baseline=None,
                      restore_best_weights=True)
    ]

    # train the model and observe the mean squared error on the development set
    history = model.fit(reviews_train, label_train,
                        validation_data=(reviews_dev, label_dev),
                        batch_size=20, epochs=200,
                        callbacks=my_callbacks, verbose=False)


    # evaluate the model on the test set
    loss, accuracy = model.evaluate(reviews_test, labels_test)
    print(f'test cross-entropy: {loss}')
    print(f'test accuracy: {accuracy}')

    return model, history


def plot_fit_hist(history):
    """
    plot Model accuracy and loss in fitting process
    :param history [callback], record of fitting process
    """
    plt.subplot(1, 2, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # Aufgabe 3.1
    w2v_dict, w2v_size = load_word2vec_embeddings('w2v_100K_emb.txt')
    reviews_dev, labels_onehot_dev = read_data_sets('./Data/rt-polarity.dev.reviews.txt',
                                                    './Data/rt-polarity.dev.labels.txt')
    reviews_test, labels_onehot_test = read_data_sets('./Data/rt-polarity.test.reviews.txt',
                                                    './Data/rt-polarity.test.labels.txt')
    reviews_train, labels_onehot_train = read_data_sets('./Data/rt-polarity.train.reviews.txt',
                                                    './Data/rt-polarity.train.labels.txt')

    reviews_dev = comput_power_mean(reviews_dev, w2v_dict, w2v_size)
    reviews_test = comput_power_mean(reviews_test, w2v_dict, w2v_size)
    reviews_train = comput_power_mean(reviews_train, w2v_dict, w2v_size)

    np.set_printoptions(precision=6)
    print('averaged word2vec embedding vector of the first review from the training data set:\n'
          f'{reviews_train[0, :300]}')
    np.set_printoptions(precision=6)
    print('the power mean word2vec embedding vector of the first review from the training data set:\n'
          f'{reviews_train[0, 900:]}')

    # Aufgabe 3.2
    model, history = MLP_modell(reviews_train, labels_onehot_train,
                            reviews_test, labels_onehot_test,
                            reviews_dev, labels_onehot_dev)
    model.save('MLP_model.h5')
