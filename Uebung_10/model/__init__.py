class Hex10Model():
    def __init__(self, params):
        self.params = params
        pass

    def train_and_predict(self):
        """
        Trains model on training data. Predicts on the test data.
        :return: Predictions results in the form [(input_1, pred_1, truth_1), (input_2, pred_2, truth_2), ...]
        """
        raise NotImplementedError