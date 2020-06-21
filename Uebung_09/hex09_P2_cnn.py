# ------------------------------------------------
#             2.1 Creating Data Splits
# ------------------------------------------------

from keras.preprocessing.sequence import pad_sequences
import random
import os

################################
#   modify path if necessary   #
input_file = 'data.txt'
################################

tmp_dir = '/tmp'
train_verbose = 0
pad_length = 300

def read_data(input_file):
    vocab = {0}
    data_x = []
    data_y = []
    with open(input_file) as f:
        for line in f:
            label, content = line.split('\t')
            content = [int(v) for v in content.split()]
            vocab.update(content)
            data_x.append(content)
            label = tuple(int(v) for v in label.split())
            data_y.append(label)

    data_x = pad_sequences(data_x, maxlen=pad_length)
    return list(zip(data_y , data_x)), vocab

data, vocab = read_data(input_file)
vocab_size = max(vocab) + 1
random.seed(42)
random.shuffle(data)
input_len = len(data)

# train_y: a list of 20-component one-hot vectors representing newsgroups
# train_y: a list of 300-component vectors where each entry corresponds to a word ID
train_y, train_x = zip(*(data[:(input_len * 8) // 10]))
dev_y, dev_x = zip(*(data[(input_len * 8) // 10: (input_len * 9) // 10]))
test_y, test_x = zip(*(data[(input_len * 9) // 10:]))




# ------------------------------------------------
#                 2.2 A Basic CNN
# ------------------------------------------------

from keras.models import Sequential, Model
from keras.layers import *

import numpy as np
train_x, train_y = np.array(train_x), np.array(train_y)
dev_x, dev_y = np.array(dev_x), np.array(dev_y)
test_x, test_y = np.array(test_x), np.array(test_y)

# Leave those unmodified and, if requested by the task, modify them locally in the specific task
batch_size = 64
embedding_dims = 100
epochs = 2
filters = 75
kernel_size = 3     # Keras uses a different definition where a kernel size of 3 means that 3 words are convolved at each step


model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))

####################################
#                                  #
#   add your implementation here   #

model.add(Conv1D(filters, kernel_size, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(20))        # 20 newsgroups
model.add(Activation('softmax'))

#                                  #
####################################

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose)
print('Accuracy of simple CNN: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])


'''
Accuracy of simple CNN: 0.769517
'''

# ------------------------------------------------
#                2.3 Early Stopping
# ------------------------------------------------

####################################
#                                  #
#   add your implementation here   #

from keras.callbacks import EarlyStopping, ModelCheckpoint

epochs = 50
patience = 2
modelname = 'task2_3.hdf5'
model_path = os.path.join('.', modelname)

model.reset_states()
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose,
                    validation_data=(dev_x, dev_y),
                    callbacks=[EarlyStopping(patience=patience, monitor='val_acc', mode='auto', verbose=1),
                               ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='auto',
                                               save_best_only=True, save_weights_only=True, verbose=1)])
model.load_weights(model_path)

print(f'Accuracy (dev) : {model.evaluate(dev_x, dev_y)[1]}')
print(f'Accuracy (test) : {model.evaluate(test_x, test_y)[1]}')

#                                  #
####################################

'''
Epoch 00001: val_acc improved from -inf to 0.81253, saving model to .\task2_3.hdf5
Epoch 00002: val_acc improved from 0.81253 to 0.83006, saving model to .\task2_3.hdf5
Epoch 00003: val_acc improved from 0.83006 to 0.83218, saving model to .\task2_3.hdf5
Epoch 00004: val_acc did not improve from 0.83218
Epoch 00005: val_acc improved from 0.83218 to 0.83324, saving model to .\task2_3.hdf5
Epoch 00006: val_acc improved from 0.83324 to 0.83537, saving model to .\task2_3.hdf5
Epoch 00007: val_acc improved from 0.83537 to 0.83590, saving model to .\task2_3.hdf5
Epoch 00008: val_acc did not improve from 0.83590
Epoch 00009: val_acc did not improve from 0.83590
Epoch 00009: early stopping
  32/1883 [..............................] - ETA: 0s
 832/1883 [============>.................] - ETA: 0s
1664/1883 [=========================>....] - ETA: 0s
1883/1883 [==============================] - 0s 62us/step
Accuracy (dev) : 0.8359001595101582
  32/1883 [..............................] - ETA: 0s
 864/1883 [============>.................] - ETA: 0s
1664/1883 [=========================>....] - ETA: 0s
1883/1883 [==============================] - 0s 63us/step
Accuracy (test) : 0.8353690920645926

'''


# ------------------------------------------------
#    2.4 Experimenting with CNN Hyperparameters
# ------------------------------------------------

####################################
#                                  #
#   add your implementation here   #

random.seed(233)

def set_random_hyperparameter():
    # randomly set hyperparameter
    # set random layer
    layers_num = random.randint(1, 4)
    # set number of Filters, Filters size(kernel_size) and Stride sizes in each layer
    layers = [(random.randint(20, 60), random.randint(1, 4), random.randint(1, 4)) for _ in range(layers_num)]
    return {"layers": layers}

def get_parameterized_model(params):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))

    for filters, kernel_size, strides in params["layers"]:
        model.add(Conv1D(filters, kernel_size, strides=strides, activation='relu'))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(20))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


search_num = 10
search_results = {}

for i in range(search_num):
    params = set_random_hyperparameter()
    model = get_parameterized_model(params)

    print(f"test {i}:\nParams (filters, kernel_size, strides):{params}")

    model_series_name = f'task2_4_test_{i}.hdf5'
    model_series_path = os.path.join('.', model_series_name)

    history = model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose,
                        validation_data=(dev_x, dev_y),
                        callbacks=[EarlyStopping(patience=patience, monitor='val_acc', mode='auto', verbose=0),
                                   ModelCheckpoint(filepath=model_series_path, monitor='val_acc', mode='auto',
                                                   save_best_only=True, save_weights_only=True, verbose=1)])

    model.load_weights(model_series_path)

    accuracy = model.evaluate(dev_x, dev_y, verbose=0)[1]
    search_results[accuracy] = (i, params)

    print(f'Test {i} Accuracy (dev) : {accuracy}')

# identify best parameter set, load corresponding best weights, evaluate on test
best_i, best_params = search_results[sorted(search_results.keys())[-1]]
best_model = get_parameterized_model(best_params)

best_model.load_weights(os.path.join('.', f'task2_4_test_{best_i}.hdf5'))

print(f'Accuracy (test) in best model {best_i}: {best_model.evaluate(test_x, test_y, verbose=0)[1]}')

#                                  #
####################################

'''
test 0:
Params (filters, kernel_size, strides):{'layers': [(53, 2, 4), (55, 2, 1)]}
Epoch 00001: val_acc improved from -inf to 0.26819, saving model to .\task2_4_test_0.hdf5
Epoch 00002: val_acc improved from 0.26819 to 0.60754, saving model to .\task2_4_test_0.hdf5
Epoch 00003: val_acc improved from 0.60754 to 0.63569, saving model to .\task2_4_test_0.hdf5
Epoch 00004: val_acc improved from 0.63569 to 0.65852, saving model to .\task2_4_test_0.hdf5
Epoch 00005: val_acc did not improve from 0.65852
Epoch 00006: val_acc improved from 0.65852 to 0.66543, saving model to .\task2_4_test_0.hdf5
Epoch 00007: val_acc improved from 0.66543 to 0.66649, saving model to .\task2_4_test_0.hdf5
Epoch 00008: val_acc improved from 0.66649 to 0.67074, saving model to .\task2_4_test_0.hdf5
Epoch 00009: val_acc did not improve from 0.67074
Epoch 00010: val_acc did not improve from 0.67074
Test 0 Accuracy (dev) : 0.6707381840342229
test 1:
Params (filters, kernel_size, strides):{'layers': [(37, 2, 1)]}
Epoch 00001: val_acc improved from -inf to 0.60754, saving model to .\task2_4_test_1.hdf5
Epoch 00002: val_acc improved from 0.60754 to 0.73500, saving model to .\task2_4_test_1.hdf5
Epoch 00003: val_acc improved from 0.73500 to 0.78120, saving model to .\task2_4_test_1.hdf5
Epoch 00004: val_acc improved from 0.78120 to 0.79501, saving model to .\task2_4_test_1.hdf5
Epoch 00005: val_acc improved from 0.79501 to 0.79873, saving model to .\task2_4_test_1.hdf5
Epoch 00006: val_acc did not improve from 0.79873
Epoch 00007: val_acc improved from 0.79873 to 0.79926, saving model to .\task2_4_test_1.hdf5
Epoch 00008: val_acc improved from 0.79926 to 0.79979, saving model to .\task2_4_test_1.hdf5
Epoch 00009: val_acc improved from 0.79979 to 0.80244, saving model to .\task2_4_test_1.hdf5
Epoch 00010: val_acc did not improve from 0.80244
Epoch 00011: val_acc improved from 0.80244 to 0.80404, saving model to .\task2_4_test_1.hdf5
Epoch 00012: val_acc did not improve from 0.80404
Epoch 00013: val_acc did not improve from 0.80404
Test 1 Accuracy (dev) : 0.8040361125546444
test 2:
Params (filters, kernel_size, strides):{'layers': [(35, 2, 3)]}
Epoch 00001: val_acc improved from -inf to 0.45459, saving model to .\task2_4_test_2.hdf5
Epoch 00002: val_acc improved from 0.45459 to 0.67233, saving model to .\task2_4_test_2.hdf5
Epoch 00003: val_acc improved from 0.67233 to 0.70207, saving model to .\task2_4_test_2.hdf5
Epoch 00004: val_acc improved from 0.70207 to 0.71482, saving model to .\task2_4_test_2.hdf5
Epoch 00005: val_acc improved from 0.71482 to 0.71641, saving model to .\task2_4_test_2.hdf5
Epoch 00006: val_acc did not improve from 0.71641
Epoch 00007: val_acc did not improve from 0.71641
Test 2 Accuracy (dev) : 0.7164099837197817
test 3:
Params (filters, kernel_size, strides):{'layers': [(50, 3, 2)]}
Epoch 00001: val_acc improved from -inf to 0.47955, saving model to .\task2_4_test_3.hdf5
Epoch 00002: val_acc improved from 0.47955 to 0.69835, saving model to .\task2_4_test_3.hdf5
Epoch 00003: val_acc improved from 0.69835 to 0.75412, saving model to .\task2_4_test_3.hdf5
Epoch 00004: val_acc improved from 0.75412 to 0.77111, saving model to .\task2_4_test_3.hdf5
Epoch 00005: val_acc improved from 0.77111 to 0.77854, saving model to .\task2_4_test_3.hdf5
Epoch 00006: val_acc did not improve from 0.77854
Epoch 00007: val_acc improved from 0.77854 to 0.78173, saving model to .\task2_4_test_3.hdf5
Epoch 00008: val_acc did not improve from 0.78173
Epoch 00009: val_acc did not improve from 0.78173
Test 3 Accuracy (dev) : 0.7817312801574305
test 4:
Params (filters, kernel_size, strides):{'layers': [(38, 4, 2)]}
Epoch 00001: val_acc improved from -inf to 0.49177, saving model to .\task2_4_test_4.hdf5
Epoch 00002: val_acc improved from 0.49177 to 0.70579, saving model to .\task2_4_test_4.hdf5
Epoch 00003: val_acc improved from 0.70579 to 0.76633, saving model to .\task2_4_test_4.hdf5
Epoch 00004: val_acc improved from 0.76633 to 0.78173, saving model to .\task2_4_test_4.hdf5
Epoch 00005: val_acc improved from 0.78173 to 0.78917, saving model to .\task2_4_test_4.hdf5
Epoch 00006: val_acc did not improve from 0.78917
Epoch 00007: val_acc improved from 0.78917 to 0.79023, saving model to .\task2_4_test_4.hdf5
Epoch 00008: val_acc did not improve from 0.79023
Epoch 00009: val_acc did not improve from 0.79023
Test 4 Accuracy (dev) : 0.7902283586533982
test 5:
Params (filters, kernel_size, strides):{'layers': [(39, 1, 4), (37, 4, 1), (45, 2, 1)]}
Epoch 00001: val_acc improved from -inf to 0.15985, saving model to .\task2_4_test_5.hdf5
Epoch 00002: val_acc improved from 0.15985 to 0.39140, saving model to .\task2_4_test_5.hdf5
Epoch 00003: val_acc improved from 0.39140 to 0.46044, saving model to .\task2_4_test_5.hdf5
Epoch 00004: val_acc improved from 0.46044 to 0.48433, saving model to .\task2_4_test_5.hdf5
Epoch 00005: val_acc improved from 0.48433 to 0.49920, saving model to .\task2_4_test_5.hdf5
Epoch 00006: val_acc improved from 0.49920 to 0.50292, saving model to .\task2_4_test_5.hdf5
Epoch 00007: val_acc improved from 0.50292 to 0.50398, saving model to .\task2_4_test_5.hdf5
Epoch 00008: val_acc did not improve from 0.50398
Epoch 00009: val_acc did not improve from 0.50398
Test 5 Accuracy (dev) : 0.5039830060000123
test 6:
Params (filters, kernel_size, strides):{'layers': [(27, 2, 2)]}
Epoch 00001: val_acc improved from -inf to 0.48115, saving model to .\task2_4_test_6.hdf5
Epoch 00002: val_acc improved from 0.48115 to 0.67021, saving model to .\task2_4_test_6.hdf5
Epoch 00003: val_acc improved from 0.67021 to 0.71375, saving model to .\task2_4_test_6.hdf5
Epoch 00004: val_acc improved from 0.71375 to 0.73022, saving model to .\task2_4_test_6.hdf5
Epoch 00005: val_acc improved from 0.73022 to 0.73712, saving model to .\task2_4_test_6.hdf5
Epoch 00006: val_acc did not improve from 0.73712
Epoch 00007: val_acc improved from 0.73712 to 0.73765, saving model to .\task2_4_test_6.hdf5
Epoch 00008: val_acc improved from 0.73765 to 0.73818, saving model to .\task2_4_test_6.hdf5
Epoch 00009: val_acc did not improve from 0.73818
Epoch 00010: val_acc improved from 0.73818 to 0.74031, saving model to .\task2_4_test_6.hdf5
Epoch 00011: val_acc did not improve from 0.74031
Epoch 00012: val_acc improved from 0.74031 to 0.74137, saving model to .\task2_4_test_6.hdf5
Epoch 00013: val_acc did not improve from 0.74137
Epoch 00014: val_acc did not improve from 0.74137
Test 6 Accuracy (dev) : 0.7413701541678297
test 7:
Params (filters, kernel_size, strides):{'layers': [(40, 4, 2), (21, 4, 1), (42, 1, 1)]}
Epoch 00001: val_acc improved from -inf to 0.31652, saving model to .\task2_4_test_7.hdf5
Epoch 00002: val_acc improved from 0.31652 to 0.67180, saving model to .\task2_4_test_7.hdf5
Epoch 00003: val_acc improved from 0.67180 to 0.71004, saving model to .\task2_4_test_7.hdf5
Epoch 00004: val_acc improved from 0.71004 to 0.71216, saving model to .\task2_4_test_7.hdf5
Epoch 00005: val_acc improved from 0.71216 to 0.73340, saving model to .\task2_4_test_7.hdf5
Epoch 00006: val_acc did not improve from 0.73340
Epoch 00007: val_acc did not improve from 0.73340
Test 7 Accuracy (dev) : 0.7334041423893836
test 8:
Params (filters, kernel_size, strides):{'layers': [(23, 4, 3), (54, 1, 4), (45, 1, 1)]}
Epoch 00001: val_acc improved from -inf to 0.15666, saving model to .\task2_4_test_8.hdf5
Epoch 00002: val_acc improved from 0.15666 to 0.43548, saving model to .\task2_4_test_8.hdf5
Epoch 00003: val_acc improved from 0.43548 to 0.46150, saving model to .\task2_4_test_8.hdf5
Epoch 00004: val_acc improved from 0.46150 to 0.48540, saving model to .\task2_4_test_8.hdf5
Epoch 00005: val_acc did not improve from 0.48540
Epoch 00006: val_acc did not improve from 0.48540
Test 8 Accuracy (dev) : 0.48539564508867594
test 9:
Params (filters, kernel_size, strides):{'layers': [(42, 4, 2), (60, 4, 3), (39, 2, 2), (51, 2, 4)]}
Epoch 00001: val_acc improved from -inf to 0.12321, saving model to .\task2_4_test_9.hdf5
Epoch 00002: val_acc improved from 0.12321 to 0.19756, saving model to .\task2_4_test_9.hdf5
Epoch 00003: val_acc improved from 0.19756 to 0.35263, saving model to .\task2_4_test_9.hdf5
Epoch 00004: val_acc improved from 0.35263 to 0.37600, saving model to .\task2_4_test_9.hdf5
Epoch 00005: val_acc improved from 0.37600 to 0.39618, saving model to .\task2_4_test_9.hdf5
Epoch 00006: val_acc improved from 0.39618 to 0.41476, saving model to .\task2_4_test_9.hdf5
Epoch 00007: val_acc did not improve from 0.41476
Epoch 00008: val_acc improved from 0.41476 to 0.42273, saving model to .\task2_4_test_9.hdf5
Epoch 00009: val_acc did not improve from 0.42273
Epoch 00010: val_acc did not improve from 0.42273
Test 9 Accuracy (dev) : 0.42272968670186123
Accuracy (test) in best model 1: 0.80669144968751

'''