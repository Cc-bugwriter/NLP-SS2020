from tensorflow import keras


def modeling(hyperparameter: dict):
    """
    build a MLP Regressor, which predicts the similarity of two sentences
    param: hyperparameter [dict], hyperparameter of MLP
    return: model
    """
    # define the model layer
    # define multi-input layers
    first_sentence_input = keras.layers.Input(shape=(300, ))
    second_sentence_input = keras.layers.Input(shape=(300, ))
    # concatenate input layer
    concatenation = keras.layers.Concatenate(axis=1)([first_sentence_input, second_sentence_input])
    # define dropout layer 1
    dropout_1 = keras.layers.Dropout(hyperparameter["Dropout_rate"][0])(concatenation)

    # append each hidden layer
    for i in range(len(hyperparameter["hidden_layer_size"])):
        if i == 0:
            dropout_interm = dropout_1

        # define dense layer i
        dense_layer_interm = keras.layers.Dense(hyperparameter["hidden_layer_size"][i],
                                                activation=hyperparameter["activation"][i])(dropout_interm)

        # define dropout layer i
        dropout_interm = keras.layers.Dropout(hyperparameter["Dropout_rate"][i])(dense_layer_interm)

    # define output layer
    output = keras.layers.Dense(1, activation=hyperparameter["activation"][-1])(dropout_interm)

    # build model
    model = keras.Model([first_sentence_input, second_sentence_input], output)

    # print model structure
    print(model.summary())
    print("Defined the model.")

    # compile the model
    model.compile(optimizer='adam', loss=keras.losses.mean_squared_logarithmic_error,
                  metrics=[keras.metrics.mean_squared_error])
    print("Compiled the model.")

    return model


if __name__ == '__main__':
    # test of MLP
    hyperparameter = {"Dropout_rate": (0.3, 0.3, 0.3),
                      "hidden_layer_size": (300, 150),
                      "activation": ('relu', 'relu', 'sigmoid')}

    MLP = modeling(hyperparameter)