"""hex06-solution"""

import nltk
import numpy as np
import tensorflow as tf


# 2 Semantic Textual Similarity
# 2.1 Data Formats
def labeled_reader(filename: str):
    """Reads the labeled dataset with the given filename."""
    scores = list()
    first_sentences = list()
    second_sentences = list()
    with open(filename, encoding='utf8') as file:
        for line in file:
            parts = line.split('\t')
            scores.append(float(parts[0]))
            first_sentences.append(parts[1].strip())
            second_sentences.append(parts[2].strip())
    return scores, first_sentences, second_sentences


def unlabeled_reader(filename: str):
    """Reads the unlabeled dataset with the given filename."""
    first_sentences = list()
    second_sentences = list()
    with open(filename, encoding='utf8') as file:
        for line in file:
            parts = line.split('\t')
            first_sentences.append(parts[0].strip())
            second_sentences.append(parts[1].strip())
    return first_sentences, second_sentences


def score_writer(filename: str, scores: [float]):
    """Writes the given list of scores to the file with the given filename."""
    with open(filename, mode='w', encoding='utf8') as file:
        file.write('\n'.join(str(score) for score in scores))


train_scores, train_first_sentences, train_second_sentences = labeled_reader('DATA/training-dataset.txt')
dev_scores, dev_first_sentences, dev_second_sentences = labeled_reader('DATA/development-dataset.txt')
test_scores, test_first_sentences, test_second_sentences = labeled_reader('DATA/test-hex06-dataset.txt')
# print(f"{train_scores[0]} {train_first_sentences[0]} {train_second_sentences[0]}")
print("Loaded the datasets.")


# 2.2 Embedding the Sentences
# load the FastText embeddings
def load_fast_text_embeddings(filename: str):
    """Loads the FastText embeddings from the file with the given filename."""
    embeddings = dict()
    embeddings_loaded = 0

    with open(filename, encoding='utf-8', newline='\n', errors='ignore') as file:
        n, d = map(int, file.readline().split())
        for line in file:

            # stop after loading 40000 embeddings
            embeddings_loaded += 1
            if embeddings_loaded >= 40000:
                break

            parts = line.rstrip().split(' ')
            embeddings[parts[0]] = np.array([float(v) for v in parts[1:]])
    return embeddings, d


token2vector, dimensions = load_fast_text_embeddings('wiki-news-300d-1M.vec')
# print(f"{'a'}: {token2vector['a']}")
print(f"Loaded the FastText embeddings with dimension {dimensions}.")


# tokenize the sentences
def tokenize_sentences(sentences: [str]):
    """Tokenizes the given sentences."""
    return [nltk.word_tokenize(sentence) for sentence in sentences]


train_first_sentences = tokenize_sentences(train_first_sentences)
train_second_sentences = tokenize_sentences(train_second_sentences)
dev_first_sentences = tokenize_sentences(dev_first_sentences)
dev_second_sentences = tokenize_sentences(dev_second_sentences)
test_first_sentences = tokenize_sentences(test_first_sentences)
test_second_sentences = tokenize_sentences(test_second_sentences)
# print(train_first_sentences[0])
print("Tokenized the sentences.")


# map the tokens to their token embedding vectors
def map_to_vectors(tokenized_sentences: [[str]]):
    """Maps the given tokenized sentences to lists of vectors."""
    # embed unknown tokens as a zero-vector
    default_token_embedding = np.zeros_like(token2vector['a'])

    # transform every sentence from a list of tokens to a list of vectors
    word_embedded_sentences = list()
    for tokenized_sentence in tokenized_sentences:
        # transform every token to its vector
        word_embedded_sentence = list()
        for token in tokenized_sentence:
            word_embedded_sentence.append(token2vector.get(token, default_token_embedding))

        word_embedded_sentences.append(word_embedded_sentence)

    return word_embedded_sentences


train_first_sentences = map_to_vectors(train_first_sentences)
train_second_sentences = map_to_vectors(train_second_sentences)
dev_first_sentences = map_to_vectors(dev_first_sentences)
dev_second_sentences = map_to_vectors(dev_second_sentences)
test_first_sentences = map_to_vectors(test_first_sentences)
test_second_sentences = map_to_vectors(test_second_sentences)
# print(train_second_sentences[0])
print("Mapped the tokens to their embedding vectors.")


# embed each sentence as the average of its word embeddings
def embed_sentences(word_embedded_sentences: [[np.array]]):
    """Calculate sentence embeddings as averages of the word embeddings."""
    embedded_sentences = list()
    for word_embedded_sentence in word_embedded_sentences:
        # calculate the power means
        s = np.zeros_like(word_embedded_sentence[0])

        for vector in word_embedded_sentence:
            s += vector

        s /= len(word_embedded_sentence)
        embedded_sentences.append(s)
    return embedded_sentences


train_first_sentences = embed_sentences(train_first_sentences)
train_second_sentences = embed_sentences(train_second_sentences)
dev_first_sentences = embed_sentences(dev_first_sentences)
dev_second_sentences = embed_sentences(dev_second_sentences)
test_first_sentences = embed_sentences(test_first_sentences)
test_second_sentences = embed_sentences(test_second_sentences)
# print(train_first_sentences[0])
print("Calculated the sentence embeddings.")


# 2.3 Scoring the Similarity

# define the model
first_sentence_input = tf.keras.layers.Input(shape=(dimensions, ))
second_sentence_input = tf.keras.layers.Input(shape=(dimensions, ))
concatenation = tf.keras.layers.Concatenate(axis=1)([first_sentence_input, second_sentence_input])
dropout_1 = tf.keras.layers.Dropout(0.3)(concatenation)
dense_layer_1 = tf.keras.layers.Dense(300, activation=tf.keras.activations.relu)(dropout_1)
dropout_2 = tf.keras.layers.Dropout(0.3)(dense_layer_1)
output = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)(dropout_2)

model = tf.keras.Model([first_sentence_input, second_sentence_input], output)
print(model.summary())
print("Defined the model.")

# compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.mean_squared_error, metrics=[tf.keras.metrics.mean_squared_error])
print("Compiled the model.")

# train the model and observe the mean squared error on the development set
model.fit([train_first_sentences, train_second_sentences], train_scores,
          validation_data=([dev_first_sentences, dev_second_sentences], dev_scores),
          batch_size=100, epochs=300)
print("Trained the model.")


# 3. Low-Level Adversarial Attacks
# evaluate the model on the test set
res = model.evaluate([test_first_sentences, test_second_sentences], test_scores)
print(res)
