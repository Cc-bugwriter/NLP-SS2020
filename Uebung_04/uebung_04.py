# Aufgabe 3.2
from gensim.models import KeyedVectors
import numpy as np
# Aufgabe 3.3
from scipy.stats import spearmanr


def read_dataset(path="./SimLex-999.txt"):
    """
    read txt data set
    :param path: [str], path of local data set (default value： "./SimLex-999.txt")
    :return:
    word_list: [list], list of tuple, which includes two compared words
    similarity_dict: [dict], dictionary of similarity in different kinds
    """
    with open(path, 'r') as file:

        # initialize return arguments list
        word_list = []
        similarity_dict = {"POS": [],
                           "SimLex999": [],
                           "conc_w1": [],
                           "conc_w2": [],
                           "concQ": [],
                           "Assoc_USF": [],
                           "SimAssoc333": [],
                           "SD_SimLex": []}

        # append element in lists (skip the first line)
        for line in file.readlines()[1:]:
            word1, word2, POS, SimLex999, conc_w1, conc_w2, \
            concQ, Assoc_USF, SimAssoc333, SD_SimLex = line.strip('\n').split('\t')

            word_list.append((word1, word2))
            similarity_dict["POS"].append(POS)
            similarity_dict["SimLex999"].append(float(SimLex999))
            similarity_dict["conc_w1"].append(float(conc_w1))
            similarity_dict["conc_w2"].append(float(conc_w2))
            similarity_dict["concQ"].append(float(concQ))
            similarity_dict["Assoc_USF"].append(float(Assoc_USF))
            similarity_dict["SimAssoc333"].append(float(SimAssoc333))
            similarity_dict["SD_SimLex"].append(float(SD_SimLex))

    return word_list, similarity_dict


def compute_distance(word_list, path='./GoogleNews-vectors-negative300.bin'):
    """
    compute Euclidiean distance between vector pairs
    :param word_list: [list], reference word pair list
    :param path: [str], path of the binary pretrained 300-dimensional word2vec embeddings from Google
    :return:
    distance: [array], Euclidiean distance of words
    """
    # initialize output distance
    dist_list = []

    # load word2vec embeddings
    # wv_from_bin = KeyedVectors.load_word2vec_format(path, binary=True)
    wv_from_bin = KeyedVectors.load_word2vec_format(path, binary=True, limit=35000)

    # compute Euclidiean distance
    for words in word_list:
        # initialize intermediate variable list
        word_vec = []
        # convert word to vector
        for i, word in enumerate(words):
            try:
                wv_from_bin.word_vec(word)
            except KeyError:
                word_vec.append(np.zeros(300, dtype='float32'))
            else:
                word_vec.append(wv_from_bin.word_vec(word))

        dist = np.linalg.norm(word_vec[0]-word_vec[1])
        dist_list.append(dist)

    return dist_list


def compute_coorelation(dist_list, similarity_dict, kind="SimLex999"):
    """
    compute Spearman’s rank correlation coefficient
    :param dist_list: [list], distance of word pairs
    :param similarity_dict: [dict], dictionary of similarity in different kinds
    :param kind: [str], metric kind of similarity
    :return:

    """
    # convert list in array
    pred = np.array(dist_list)
    true = np.array(similarity_dict[kind])

    # compute Spearman’s rank correlation coefficient
    rho = spearmanr(pred, true)

    return rho


if __name__ == '__main__':
    SimLex_path = "./SimLex-999.txt"
    word_list, similarity_dict = read_dataset(path=SimLex_path)

    GoogleNews_path = './GoogleNews-vectors-negative300.bin'
    dist_list = compute_distance(word_list, path=GoogleNews_path)

    ref_kind = "SimLex999"
    rho = compute_coorelation(dist_list, similarity_dict, kind=ref_kind)

    print("the Spearman’s rank correlation coefficient :")
    print(rho)
    if rho.correlation > 0:
        print(f"the prediction has positive correlation with {ref_kind}")
    else:
        print(f"the prediction has negative correlation with {ref_kind}")


