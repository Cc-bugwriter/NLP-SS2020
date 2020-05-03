import csv  # Aufgabe 2.1
import numpy as np  # Aufgabe 2.2


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


class Perceptron(object):

    def __init__(self):
        """
        Initialization of Preceptron with Training Data set
        """
        self.Weight = np.random.normal(0, 1, (100 + 1, 1))  # Input with bias
        self.alfa = 0.01  # learning rate
        self.size = 10  # size of mini batch
        self.epochs = 100  # epoch for iteration

    def loss_function(self, X, y):
        """
        Return square-loss of simultaneous perception
        """
        (_, z) = weighting(X, self.Weight)  # intermediate variables
        a = sigmoid(z)  # activation function
        loss = sum(np.square(a - y))
        return loss


    def mini_batch(self, X, seed, size):
        """
        Return a random subset of the whole training data
        :param X: [narray], whole training data or target data
        :param seed: int, random seed for shuffle
        :param size: int, the size of batch subset
        :return: [list of narray], all random subsets
        """
        np.random.seed(seed)
        np.random.shuffle(X)
        all_batch = [X[k: k + size, :] for k in range(0, X.shape[0], size)]
        return all_batch


    def SGD(self, X, y, alfa=0.01, size=10, epoch=100):
        """
        Updating the weight matrix base on
        mini-batch stochastic gradient descent
        :param X: [narray], Input data set
        :param alfa: float, learning rate, (default value: 0.01)
        :param size: int, the size of batch subset, (default value: 10)
        :param epoch: int, iteration times, (default value: 100)
        :return:
        """
        for iteration in range(epoch):
            seed = np.random.randint(10)  # initialize the seed of mini batch
            X_all_batch = self.mini_batch(X, seed, size)
            y_all_batch = self.mini_batch(y, seed, size)
            for i, X_mini in enumerate(X_all_batch):
                (X_mini_bias, z_mini) = weighting(X_mini, self.Weight)
                a_mini = sigmoid(z_mini)
                a_prime_mini = sigmoid_prime(z_mini)
                y_mini = y_all_batch[i]
                delta_w = -alfa / size * np.sum(np.dot((a_mini - y_mini).T, a_prime_mini) * X_mini_bias.T, axis=1, keepdims=True)

                self.Weight += delta_w


    def evaluation(self, X, y):
        """
        Evaluate accuracy of predict
        :param X: [narray], Input data
        :param y: [narray], Target data
        :return acc: float, accuracy of predict
        """
        (_, z) = weighting(X, self.Weight)  # intermediate variables
        a = sigmoid(z)  # activation function
        y_pred = predict(a) # predict classification
        acc = sum(np.abs(y_pred == y))/y.size
        return acc


## Miscellaneous functions
def weighting(X, theta):
    """
    Calculate the intermediate variables of activation function
    :param theta: [narray], weight vector (n+1*1)
    :param X: [narray], Input Matrix
    :return X_bias: [narray], Input Matrix with bias term
    :return z: [narray], intermediate variables of activation
    """
    try:
        np.ones(X.shape[0])
    except IndexError:
        bias = np.ones(1)
    else:
        bias = np.ones((X.shape[0],1))

    X_bias = np.append(bias, X, axis=1) # add a bias column in feature
    z = np.dot(X_bias, theta)
    return X_bias, z


def sigmoid(z):
    """
    The sigmoid function
    :param z: [narray], intermediate variables
    :return: [narray], probability of intermediate variables
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    :param z: [narray], intermediate variables
    :return: [narray], probability density of intermediate variables
    """
    return sigmoid(z) * (1.0 - sigmoid(z))


def predict(a):
    """
    Predict whether the output positive or negative
    :param a: activation output
    :return y_pred:
    """
    a[a >= 0.5] = 1
    a[a < 0.5] = 0
    y_pred = a
    return y_pred

def regularization(X):
    """
    Regularization input data so that speed up convergence of loss function
    :param X:[narray], input data
    :return X: [narray], regularized input data
    :return feature_mean: [list], mean of input data
    :return feature_std: [list], std of input data
    """
    feature_mean = []
    feature_std = []
    for i in range(X.shape[1]):
        feature_mean.append([X[i].mean()])
        feature_std.append([X[i].std()])
        X[i] -= X[i].mean()
        X[i] /= X[i].std()
    return X, feature_mean, feature_std

def map_regularization(X, feature_mean, feature_std):
    """
    Map the regularization of input data (for dev and test data set)
    :param X: [narray], data set (dev and test data set)
    :param feature_mean: [list], mean of input data (training)
    :param feature_std: [list], std of input data (training)
    :return X: [narray],  regularized data set
    """
    for i in range(X.shape[1]):
        X[i] -= feature_mean[i]
        X[i] /= feature_std[i]
    return X

if __name__ == '__main__':

    # Load data
    (X_train, Y_train) = Dataset_reader(name='train')
    (X_dev, Y_dev) = Dataset_reader(name='dev')
    (X_test, Y_test) = Dataset_reader(name='test')

    # regularization
    (X_train, feature_mean, feature_std) = regularization(X_train)
    X_dev = map_regularization(X_dev, feature_mean, feature_std)
    X_test = map_regularization(X_test, feature_mean, feature_std)

    Perceptron = Perceptron()
    # before Training
    Loss_before = Perceptron.loss_function(X=X_test, y=Y_test)
    acc_before = Perceptron.evaluation(X=X_test, y=Y_test)
    # Training
    Perceptron.SGD(X=X_train, y=Y_train, alfa=0.003, size=10, epoch=100)
    # after Training
    Loss_after = Perceptron.loss_function(X=X_test, y=Y_test)
    acc_after = Perceptron.evaluation(X=X_test, y=Y_test)
    print('---------------')
    print('accuracy before : %f' % acc_before)
    print('Loss before : %f' % Loss_before)
    print('---------------')
    print('accuracy after : %f' % acc_after)
    print('Loss after : %f' % Loss_after)
