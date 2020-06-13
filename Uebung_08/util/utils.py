import os
import re
import numpy as np

def get_pretrained_embeddings(filename, embeddings_path="embeddings"):
    # extract the embedding dimension from the filename (simpler solutions are imaginable)
    embedding_dimension = int(re.search(r"\d+(?=d)", filename).group(0))

    with open(os.path.join(embeddings_path, filename), 'r', encoding="utf8") as file:
        # find out what the number of lines, i.e. the number of words with known embeddings is
        vocabulary_size = sum([1 for line in file])
        # reset the file pointer to the beginning of the file
        file.seek(0)

        # +=2 because we reserve one entry in the embedding matrix for OOV tokens and another one for our padding token
        vocabulary_size += 2
        embedding_matrix = np.empty((vocabulary_size, embedding_dimension), dtype=np.float32)
        word_to_embedding = {}

        embedding_matrix[0] = np.zeros(embedding_dimension)  # index 0 for __PADDING__
        embedding_matrix[1] = np.random.randn(embedding_dimension)  # index 1 for OOV words
        word_to_embedding["__PADDING__"] = 0
        word_to_embedding["__OOV__"] = 1

        # starting with index 2, we enter the regular words
        for i, line in enumerate(file, 2):
            parts = line.split()
            word_to_embedding[parts[0]] = i
            embedding_matrix[i] = np.array(parts[1:], dtype=np.float32)

    return embedding_matrix, word_to_embedding


# dict to keep track of which tags are in which column in the files
task_to_column = {"pos": 1, "chunk": 2, "ner": 3}


def load_dataset(filename, data_path="data"):
    """ Load full dataset (all tasks)
    """
    sentences = []
    with open(os.path.join(data_path, filename), 'r') as file:
        file = iter(file)
        sentence = []
        while True:
            try:
                line = next(file)
            except StopIteration:
                break

            # end-of-sentence is encoded as an empty (falsy) line
            if not line.strip():
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(line.split())
    return sentences


def get_task_data(task, dataset):
    ''' Given a task, filters out all other task labels from the given dataset
    '''
    xs = []
    ys = []
    for sentence in dataset:
        x = [tup[0] for tup in sentence]
        xs.append(x)

        y = [tup[task_to_column[task]] for tup in sentence]
        ys.append(y)
    return xs, ys



def get_index_dict(input_data):
    """
    Create index - word/label dict from list input
    @params : List of lists
    @returns : Index - Word/label dictionary
    """
    result = dict()
    vocab = set()
    i = 1
    # Flatten list and get indices
    for element in [word for sentence in input_data for word in sentence]:
        if element not in vocab:
            result[i]=element
            i+=1
            vocab.add(element)
    return result


def get_prediction_results(data, labels, predict_prob, vocab_dict, label_dict):
    """
    Get the prediction and the true labels with the words in conll format
    @params : data - unpadded test data
    @params : labels - unpadded test labels
    @params : predict_prob - the predicted probabilities
    @params : vocab_dict - vocabulary dict
    @params : label_dict - label dict
    @returns : result - Resulting textual predictions in conll format:
            line_number(int)    word(str)   predict_label(str)  true_label(str)
    """
    result = []
    for document, doc_labels, doc_preds in zip(data, labels, predict_prob):
        document_result = []
        for word, word_label, word_pred in zip(document, doc_labels, doc_preds):
            # Get index of most probable label:
            index = word_pred.tolist().index(max(word_pred))
            # Conll format: line_number(int)    word(str)   predict_label(str)  true_label(str)
            document_result.append((vocab_dict[word],label_dict[index],label_dict[word_label]))
        result.append(document_result)
    return result