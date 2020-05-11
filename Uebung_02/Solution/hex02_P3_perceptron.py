import numpy as np
import os


# ------------------------------------------------
#               3.1 DATASET READER
# ------------------------------------------------

label_pos = 1
label_neg = 0

data_dim = 100
data_dim_with_bias = data_dim + 1

def read_dataset(src):
    """Reads a dataset from the specified path and returns input vectors and labels in an array of shape (n, 101)."""
    with open(src, 'r') as src_file:
        # preallocate memory for the data
        num_lines = sum(1 for line in src_file)
        data = np.empty((num_lines, data_dim_with_bias), dtype=np.float16)
        labels = np.empty((num_lines, 1), dtype=np.float16)

        # reset the file pointer to the beginning of the file
        src_file.seek(0)
        for i, line in enumerate(src_file):
            _, str_label, str_vec = line.split('\t')
            labels[i] = label_pos if str_label.split('=')[1] == "POS" else label_neg
            data[i,:data_dim] = [float(f) for f in str_vec.split()]
            data[i,data_dim] = 1
    return data, labels

def get_dataset(src_folder, name="train"):
    path = os.path.join(src_folder, "rt-polarity.{}.vecs".format(name))
    return read_dataset(path)

def get_random_batches(X, y, batch_size):
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]
    # when using array_split for 100 datapoints and batch size 33 one would get batches [33, 33, 33, 1]
    X_batches = np.array_split(X, len(y)//batch_size)
    y_batches = np.array_split(y, len(y)//batch_size)
    return X_batches, y_batches

# load the data
train_x, train_y, = get_dataset("DATA", "train")
dev_x, dev_y = get_dataset("DATA", "dev")
test_x, test_y = get_dataset("DATA", "test")



# ------------------------------------------------
#           3.2 a) NUMPY IMPLEMENTATION
# ------------------------------------------------

def sigmoid(v):
      return 1.0 / (1+np.exp(-v))

def train_np(train_x, train_y, w, epochs, batch_size, learning_rate):

    def epoch(train_x, train_y, w_init, batch_size, learning_rate):
        w = w_init
        train_x_batches, train_y_batches = get_random_batches(train_x, train_y, batch_size)
        for X_batch, y_batch in zip(train_x_batches, train_y_batches):
            grad = 0
            for x, y in zip(X_batch, y_batch):
                sigxw = sigmoid(np.dot(x, w))
                grad += (sigxw-y) * sigxw * (1 - sigxw) * x
            # average the gradient
            grad /= len(y_batch)
            w -= learning_rate * grad
        return w

    eval_every_ith_epoch = 20
    for i in range(epochs):
        w = epoch(train_x, train_y, w, batch_size, learning_rate)
    
        # compute the loss every i epochs
        if not i%eval_every_ith_epoch:
            dev_loss, _ = eval_np(dev_x, dev_y, w)
            train_loss, _ = eval_np(train_x, train_y, w)
            print("Epoch {}, Train Loss: {}, Dev Loss: {}".format(i, train_loss, dev_loss))

    return w

def eval_np(eval_x, eval_y, w):
    predictions = [sigmoid(np.dot(x, w)) for x in eval_x]
    predictions_discrete = [np.rint(pred) for pred in predictions]
    
    n = len(eval_y)    
    mean_square_loss = sum([(pred - y)**2 for pred, y in zip(predictions, eval_y)])/n
    accuracy = sum([pred == y for pred, y in zip(predictions_discrete, eval_y)])/n

    return mean_square_loss, accuracy

def run_numpy(epochs=100, batch_size=10, learning_rate=0.01, test=False):
    np.random.seed(seed=42)
    np.seterr(all='ignore')

    w_init = np.random.normal(0, 1, (data_dim_with_bias))
    w = train_np(train_x, train_y, w_init, epochs, batch_size, learning_rate)

    # print results on dev (and test)
    loss, accuracy = eval_np(dev_x, dev_y, w)
    print("Loss on dev after {} epochs: {}, accuracy: {}".format(epochs, loss, accuracy))

    if test:
        loss, accuracy = eval_np(test_x, test_y, w)
        print("Loss on test after {} epochs: {}, accuracy: {}".format(epochs, loss, accuracy))



# ------------------------------------------------
#           3.2 b) EXPERIMENT RESULTS
# ------------------------------------------------

# initial parameters
print("Running with initial parameters...")
run_numpy()
# console output:
#   Running with initial parameters...
#   Epoch 0, Train Loss: [0.2743837], Dev Loss: [0.4724828]
#   Epoch 20, Train Loss: [0.2743837], Dev Loss: [0.38649157]
#   Epoch 40, Train Loss: [0.2743837], Dev Loss: [0.34208882]
#   Epoch 60, Train Loss: [0.2743837], Dev Loss: [0.31910568]
#   Epoch 80, Train Loss: [0.2743837], Dev Loss: [0.31066293]
#   Loss on dev after 100 epochs: [0.3056598], accuracy: [0.69043152]
print("\n\n")

# slightly better parameter set, determined by manual experimentation on the dev set (not shown)
print("Running more optimal parameters...")
run_numpy(epochs=300, batch_size=25, learning_rate=0.03, test=True)
# console output:
#   Running more optimal parameters...
#   Epoch 0, Train Loss: [0.2743837], Dev Loss: [0.46810508]
#   Epoch 20, Train Loss: [0.2743837], Dev Loss: [0.38555348]
#   Epoch 40, Train Loss: [0.2743837], Dev Loss: [0.38117573]
#   Epoch 60, Train Loss: [0.2743837], Dev Loss: [0.33020636]
#   Epoch 80, Train Loss: [0.2743837], Dev Loss: [0.31394622]
#   Epoch 100, Train Loss: [0.2743837], Dev Loss: [0.30065665]
#   Epoch 120, Train Loss: [0.2743837], Dev Loss: [0.30284554]
#   Epoch 140, Train Loss: [0.2743837], Dev Loss: [0.30550343]
#   Epoch 160, Train Loss: [0.2743837], Dev Loss: [0.28689805]
#   Epoch 180, Train Loss: [0.27264202], Dev Loss: [0.29409006]
#   Epoch 200, Train Loss: [0.2721061], Dev Loss: [0.2853346]
#   Epoch 220, Train Loss: [0.2743837], Dev Loss: [0.28361475]
#   Epoch 240, Train Loss: [0.26393354], Dev Loss: [0.28298938]
#   Epoch 260, Train Loss: [0.26098606], Dev Loss: [0.28173858]
#   Epoch 280, Train Loss: [0.26567525], Dev Loss: [0.27939337]
#   Loss on dev after 300 epochs: [0.28064415], accuracy: [0.71607255]
#   Loss on test after 300 epochs: [0.3005003], accuracy: [0.69668543]