import numpy as np
np.random.seed(42)

from . import Hex10Model
from .data_reader import load_dataset

from keras.models import Model
from keras.layers import Input, LSTM, Dense


class Hex10Seq2Seq(Hex10Model):
    def __init__(self, params):
        super(Hex10Seq2Seq, self).__init__(params)
        self.batch_size = params["batch_size"]
        self.hidden_units = params["hidden_units"]
        self.epochs = params["epochs"]


    def train_and_predict(self):
        """
        Trains model on training data. Predicts on the test data.
        :return: Predictions results in the form [(input_1, pred_1, truth_1), (input_2, pred_2, truth_2), ...]
        """

        # Load data: [[token11, token12, ...],[token21,token22,...]]
        # and label: [[label11, label12, ...],[label21,label22,...]]
        X_train_data, y_train_data = load_dataset("train.dat", seq2seq=True)
        X_dev_data, y_dev_data = load_dataset("dev.dat", seq2seq=True)
        X_test_data, y_test_data = load_dataset("test.dat", seq2seq=True)

        ####################################
        #                                  #
        #   add your implementation here   #
        # all the code borrows from https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
        def find_length(X):
            """
            find the appropriate length of tokens, seq and the index of token
            X: [list], input word's list
            """
            # find an unordered and non-repetitive element set of characters
            characters = set("".join(X))
            # sort (prepare to index tokens)
            characters = sorted(characters)
            # assign index dict from sorted characters
            index_tokens = {char: i for i, char in enumerate(characters)}
            # assign length of all characters
            len_tokens = len(characters)
            # assign max length of each character
            len_seq_max = max([len(character) for character in X])

            return len_tokens, len_seq_max, index_tokens

        num_encoder_tokens, max_encoder_seq_length, input_token_index = find_length(X_train_data + X_dev_data)
        num_decoder_tokens, max_decoder_seq_length, target_token_index = find_length(y_train_data + y_dev_data)

        # cause of OOV in test
        num_encoder_tokens += 1
        print('Number of samples:', len(X_train_data + X_dev_data))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        encoder_input_data = np.zeros(
            (len(X_train_data), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(X_train_data), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(X_train_data), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        # prepare for test
        encoder_input_data_test = np.zeros(
            (len(X_test_data), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')

        # assign OOV dict in index dict
        input_token_index.update({'OOV': len(input_token_index)})

        for i, (input_text, target_text) in enumerate(zip(X_train_data, y_train_data)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            # encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
            # decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
            # decoder_target_data[i, t:, target_token_index[' ']] = 1.

        for i, input_text in enumerate(X_test_data):
            for t, char in enumerate(input_text):
                # set OOV index hot if exist
                if char in input_token_index:
                    encoder_input_data_test[i, t, input_token_index[char]] = 1.
                else:
                    encoder_input_data_test[i, t, input_token_index['OOV']] = 1.

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(self.hidden_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.hidden_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training (instead of spilt)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs)
                  # validation_split=0.2)
        # Save model
        model.save('s2s.h5')

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.hidden_units,))
        decoder_state_input_c = Input(shape=(self.hidden_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0, target_token_index['^']] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value)

                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sampled_token_index]
                decoded_sentence += sampled_char

                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '$' or
                        len(decoded_sentence) > max_decoder_seq_length):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, sampled_token_index] = 1.

                # Update states
                states_value = [h, c]

            return decoded_sentence

        # predict (on test set)
        predictions = []

        print('run prediction')
        for seq_index in range(len(X_test_data)):
            decoded_sentence = decode_sequence(encoder_input_data_test[seq_index: seq_index+1])
            predictions.append((X_test_data[seq_index], decoded_sentence[:-1], y_test_data[seq_index][1:-1]))

        return predictions
        #                                  #
        ####################################


if __name__ == "__main__":
    # test
    X_train_data, y_train_data = load_dataset("train.dat", data_path="../data", seq2seq=True)
    X_dev_data, y_dev_data = load_dataset("dev.dat", seq2seq=True)

    input_characters = sorted(set("".join(X_train_data + X_dev_data)))


