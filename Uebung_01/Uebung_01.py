#### Libraries
import numpy as np

class KNN_Muster(object):

    def __init__(self):
        self.weights = np.array([-1.0, 1.0])
        self.X = np.array([[-1.28, 0.09],[0.17, 0.39],[1.36, 0.46],[-0.51, -0.32]])
        self.y = np.array([0, 1, 1, 0])
        self.X_test = np.array([[-0.5, -1.0], [0.75, 0.25]])
        self.y_test = np.array([0, 1])
        self.alfa = 1.0 # learning rate
        
    # Aufgabe 2
    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    # Aufgabe 3_1
    def feedforward(self):
        """Return the output[list] of the activation."""
        # initialize
        a = [] # activation
        a_p = [] # activation derivative
        # simultaneous calculate activation and it's derivative
        for x in self.X:
            element = self.sigmoid(np.dot(self.weights, x))
            a.append(element)
            
            element = self.sigmoid_prime(np.dot(self.weights, x))
            a_p.append(element)
        return np.array(a), np.array(a_p)

    def update_weights(self):
        """Return the updated weight vector."""
        activation, activation_p = self.feedforward()
        # initialize delta_weights
        delta_w = np.zeros(2)
        # simultaneous calculate delta_weights
        for i, element in enumerate(self.y):
            delta_w += (activation[i]-element)*(activation_p[i])*self.X[i]
        # update weight
        self.weights -= self.alfa*delta_w

    # Aufgabe 3_2
    def evaluate(self):
        """Return the result of Loss function."""
        # initialize delta_weights
        Loss = 0
        for i, x_test in enumerate(self.X_test):
            Loss += (self.sigmoid(np.dot(self.weights,x_test))-self.y_test[i])**2
        return Loss
                               
if __name__ == '__main__':
    
    KNN = KNN_Muster()
    Loss_before = KNN.evaluate()
    
    # Training (200 iteration)
    iteration = 500
    for i in range(iteration):
        KNN.update_weights()
        
    Loss_after = KNN.evaluate()
    
    print('iteration: %f' %iteration)
    print('Weight Vector:')
    print(KNN.weights)
    print('Square loss before Training: %f' %Loss_before)
    print('Square loss after Training: %f' %Loss_after)
