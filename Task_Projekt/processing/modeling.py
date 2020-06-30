from tensorflow import keras
from preprocessing import convert_vector

def modeling_MLP(hyperparameter: dict):
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

def modeling_LSTM_MLP(hyperparameter: dict):
    """
       build a MLP Regressor, which predicts the similarity of two sentences
       param: hyperparameter [dict], hyperparameter of MLP
       return: model
    """
    # assign embedding_matrix
    embedding_matrix, _ = convert_vector.get_pretrained_embeddings()

    # assign max_sentence_length
    max_sentence_length = 56

    embedding_layer = keras.layers.Embedding(len(embedding_matrix),
                                                len(embedding_matrix[0]),
                                                weights=[embedding_matrix],
                                                input_length=max_sentence_length,
                                                trainable=False)

    lstm_layer = keras.layers.LSTM(300)

    sequence_1_input = keras.layers.Input(shape=(max_sentence_length,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    y1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = keras.layers.Input(shape=(max_sentence_length,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y2 = lstm_layer(embedded_sequences_2)

    merged = keras.layers.Concatenate(axis=1)([y1, y2])
    merged = keras.layers.Dropout(hyperparameter["Dropout_rate"][0])(merged)
    merged = keras.layers.BatchNormalization()(merged)

    # append each hidden layer
    for i in range(len(hyperparameter["hidden_layer_size"])):
        # define dense layer i
        merged = keras.layers.Dense(hyperparameter["hidden_layer_size"][i],
                                    activation=hyperparameter["activation"][i])(merged)

        # define dropout layer i
        merged = keras.layers.Dropout(hyperparameter["Dropout_rate"][i])(merged)
        merged = keras.layers.BatchNormalization()(merged)

    output = keras.layers.Dense(1, activation=hyperparameter["activation"][-1])(merged)

    model = keras.Model(inputs=[sequence_1_input, sequence_2_input], outputs=output)

    model.compile(optimizer='adam', loss=keras.losses.mean_squared_error,
                  metrics=[keras.metrics.mean_squared_error])
    model.summary()
    return model


if __name__ == '__main__':
    # test of MLP
    hyperparameter = {"Dropout_rate": (0.3, 0.3, 0.3),
                      "hidden_layer_size": (300, 150),
                      "activation": ('relu', 'relu', 'sigmoid')}

    MLP = modeling_MLP(hyperparameter)
    l_MLP = modeling_LSTM_MLP(hyperparameter)

