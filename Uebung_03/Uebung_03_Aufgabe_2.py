import csv  # Aufgabe 2.1
import numpy as np
import tensorflow as tf  # Aufgabe 2.2, 2.3


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
    MLP Preceptron base on Tensorflow
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
        self.target = target_set
        # assign hyper parameter
        self.hyper_parameter = hyper_parameters
        # define input interface of tf
        self.input_layer = tf.compat.v1.placeholder(tf.float64, shape=(None, input_set.shape[1]+1))
        # define target interface of tf
        self.target_layer = tf.compat.v1.placeholder(tf.float64, shape=(None, target_set.shape[1]))

        # create network of layers (list)
        network = [self.input_layer]  # initialize

        for layer_parameters in hyper_parameters["hidden_layers"]:
            neuron_units, activation, regularizer = layer_parameters
            print(neuron_units)
            # assign layer parameter
            layer = tf.keras.layers.Dense(neuron_units,
                                          activation=activation,
                                          use_bias=True,
                                          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                          kernel_initializer=tf.random_normal_initializer())
            network.append(layer)

        # assign prediction layer
        self.preception_layer = tf.keras.layers.Dense(target_set.shape[1],
                                                      hyper_parameters["output_activation"])
        # assign evaluation
        self.loss = hyper_parameters["loss_function"](self.target_layer, self.preception_layer)
        self.optimizer = hyper_parameters["optimizer"](hyper_parameters["learning_rate"]).minimize(self.loss)
        self.accuracy, _ = tf.metrics.accuracy(self.preception_layer, self.target_layer)


    def processing(self):
        """
        train MLP model with Training data set
        print current training progress
        :return: sess:[Session], instance of computing environment
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # run iteration
            for i in range(self.hyper_parameters["epochs"]):
                # prove mini batch
                if "batch_size" in self.hyper_parameters:
                    # implement mini batch
                    all_batch_input, all_batch_target = get_mini_batch(self.input, self.target,
                                                                       self.hyper_parameters["batch_size"])
                    for input_batch, target_batch in zip(all_batch_input, all_batch_target):
                        sess.run(self.optimizer,
                                 feed_dict={self.input_layer: input_batch, self.target_layer: target_batch})
                else:
                    # without mini batch
                    sess.run(self.optimizer, feed_dict={self.input_layer: self.input, self.target_layer: self.target})

                # print training progress
                print("training process in {} %".format(i*100/float(self.hyper_parameters["epochs"])), flush=True)

            return sess


    def evaluation(self, sess, test_input, test_target):
        """
        evaluate result of MLP Preceptron
        :param sess:[Session], instance of computing environment
        :param test_input: [narray], input of test data set
        :param test_target: [narray], target of test data set
        :return: preception [float], loss function
        :return: accuracy [float], accuracy of classifier
        :return: loss [float], loss function
        """
        # predict classification
        preception = sess.run([self.preception_layer],
                               feed_dict={self.input_layer: test_input, self.target_layer: test_target})

        # compute accuracy
        accuracy = sess.run([self.accuracy],
                            feed_dict={self.input_layer: test_input, self.target_layer: test_target})

        # compute loss
        loss = sess.run([self.loss],
                               feed_dict={self.input_layer: test_input, self.target_layer: test_target})


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
    # parameters for task 2.2
    hyper_parameters = {"batch_size": 10,
                        "learning_rate": 0.01,
                        "epochs": 100,
                        "hidden_layers": [(50, tf.tanh, False), (50, tf.tanh, False)],
                        "output_activation": tf.tanh,
                        "loss_function": tf.compat.v1.losses.mean_squared_error,
                        "optimizer": tf.compat.v1.train.GradientDescentOptimizer}

    # Load data
    (X_train, Y_train) = Dataset_reader(name='train')
    (X_dev, Y_dev) = Dataset_reader(name='dev')
    (X_test, Y_test) = Dataset_reader(name='test')

    # modeling
    MLP = MLP_Preceptron(X_train, Y_train, hyper_parameters)
    MLP_sess = MLP.processing(hyper_parameters)
    preception, accuracy, loss =MLP.evaluation(MLP_sess, X_test, Y_test)