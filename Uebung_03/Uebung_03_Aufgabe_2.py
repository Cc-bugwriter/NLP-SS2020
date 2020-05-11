import csv  # Aufgabe 2.1
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import SGD


def Dataset_reader(path='DATA/rt-polarity', name='train', type='vecs'):
    """
    read the data set
    and return Input and Target of Training Network
    :param path: [str], the path of vecs data (default value : 'DATA/rt-polarity')
    :param name: [str], data to import ('train' ,'dev' or 'test') (default value : 'train')
    :param type: [str], data type (default value : 'vecs')
    :return label: [narray], Target of Training Network
    :return vector: [narray], Input of Training Network
    """
    data = f"{path}.{name}.{type}"
    with open(data, newline='') as training_data:
        data_reader = csv.reader(training_data, delimiter='\t')
        label = []
        vector = []
        for row in data_reader:
            if row[1] == 'label=NEG':
                label.append([0])
            else:
                label.append([1])

            # transform string to float
            list_value = [float(i) for i in row[2].split()]

            vector_m = []
            for element in list_value[:100]:
                # 100 features per example
                vector_m.append(element)

            vector.append(vector_m)
    label = np.array(label, dtype='double')
    vector = np.array(vector, dtype='double')

    return vector, label


class MLP_Preceptron():
    """
    MLP Preceptron base on Keras
    """
    def __init__(self, input_set, target_set, hyper_parameters):
        """
        initialize model
        :param input_set
        :param target_set
        :param hyper_parameters: [dict], define the structure of MLP
        """
        # assign input data set
        self.input = input_set
        # assign target data set
        self.target = to_categorical(target_set)
        # assign hyper parameter
        self.hyper_parameter = hyper_parameters
        # define input interface of tf
        input_layer_shape = (input_set.shape[1],)

        # create model of Sequence
        self.model = Sequential()

        # assign hidden layer parameter
        for i, layer_parameters in enumerate(hyper_parameters["hidden_layers"]):
            neuron_units, activation, regularizer = layer_parameters
            # define input shape
            if i == 0:
                # define input layer
                layer = Dense(neuron_units,
                              activation=activation,
                              use_bias=True,
                              kernel_regularizer=regularizer,
                              kernel_initializer='random_uniform',
                              input_shape=input_layer_shape)
            else:
                layer = Dense(neuron_units,
                              activation=activation,
                              use_bias=True,
                              kernel_regularizer=regularizer,
                              kernel_initializer='random_uniform')

            # add layer on model
            self.model.add(layer)

        # define prediction layer
        preception_layer = Dense(target_set.shape[1]+1, activation=hyper_parameters["output_activation"])
        # add output layer on model
        self.model.add(preception_layer)

        # define model compile
        optimizer = SGD(lr=hyper_parameters["learning_rate"])
        self.model.compile(optimizer=optimizer, loss=hyper_parameters["loss_function"], metrics=['accuracy'])

        # define early_stopping_monitor
        self.early_stopping_monitor = EarlyStopping(patience=2)


    def processing(self):
        """
        train MLP model with Training data set
        """
        self.model.fit(self.input, self.target, epochs=self.hyper_parameter["epochs"],
                       batch_size=self.hyper_parameter["batch_size"],
                       callbacks=[self.early_stopping_monitor], verbose=False)


    def evaluation(self, test_input, test_target):
        """
        evaluate result of MLP Preceptron
        :param test_input: [narray], input of test data set
        :param test_target: [narray], target of test data set
        :return: preception [array], preception of MLP
        :return: accuracy [float], accuracy of classifier
        :return: loss [float], loss function
        """
        # predict classification
        prediction = self.model.predict(test_input)  # direct callback model
        prediction = prediction[:,1]

        # compute accuracy
        accuracy_computer = tf.keras.metrics.Accuracy()  # initial accuracy metrics
        accuracy_computer.update_state(test_target, prediction)  # compute
        accuracy = accuracy_computer.result().numpy()  # convert result in array

        # compute loss
        if self.hyper_parameter["loss_function"] == 'mean_squared_error':
            loss_computer = tf.keras.losses.MeanSquaredError()
            loss = loss_computer(test_target, prediction).numpy()

        return preception, accuracy, loss


## Miscellaneous functions
def get_mini_batch(input_set, target_set, batch_size, seed=233):
    """
    return a random mini_batch of the whole training data
    :param input_set: [narray], whole training data set
    :param target_set: [narray], whole target data set
    :param batch_size: int, the size of batch subset
    :param seed: int, random seed for shuffle (default value: 233)
    :return: [list of narray], all random mini_batch dataset
    """
    np.random.seed(seed)
    np.random.shuffle(input_set)
    np.random.shuffle(target_set)
    all_batch_input = [input_set[k: k + batch_size, :] for k in range(0, input_set.shape[0], batch_size)]
    all_batch_target = [target_set[k: k + batch_size, :] for k in range(0, target_set.shape[0], batch_size)]
    return all_batch_input, all_batch_target


if __name__ == '__main__':

    # assign hyper parameters
    hyper_parameters = {"batch_size": 10,
                        "learning_rate": 0.003,
                        "epochs": 800,
                        "hidden_layers": [(80, 'relu', None), (50, 'relu', None)],
                        "output_activation": 'softmax',
                        "loss_function": 'mean_squared_error',
                        "optimizer": 'sgd'}

    # Load data
    (X_train, Y_train) = Dataset_reader(name='train')
    (X_dev, Y_dev) = Dataset_reader(name='dev')
    (X_test, Y_test) = Dataset_reader(name='test')

    # modeling
    MLP = MLP_Preceptron(X_train, Y_train, hyper_parameters)

    MLP.processing()

    preception, accuracy, loss =MLP.evaluation(X_test, Y_test)