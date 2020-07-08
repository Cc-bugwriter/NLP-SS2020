import numpy as np
np.random.seed(42)

import os

def load_dataset(filename, data_path="./data", seq2seq=False):
    inflected_words = []
    lemmata = []

    with open(os.path.join(data_path, filename), 'r') as lines:
        # lists of characters of the inflected word and the lemma
        inflec = []
        lemma = []

        for line in lines:
            # empty line -> a word ends
            if not line.strip():
                if seq2seq:
                    # pass
                    ##########################################
                    #                                        #
                    #   maybe add your implementation here   #
                    # convert character list to str
                    inflec = ''.join(inflec)
                    # add the start character stop character to lemma
                    lemma = ''.join(['^', *lemma, '$'])
                    #                                        #
                    ##########################################

                # store assembled inputs
                inflected_words.append(inflec)
                lemmata.append(lemma)
                inflec = []
                lemma = []
                continue

            inflec_char, lemma_char = line.strip().split('\t')
            inflec.append(inflec_char)

            if seq2seq:
                # pass
                ##########################################
                #                                        #
                #   maybe add your implementation here   #
                if "_MYJOIN_" in lemma_char:
                    lemma.extend(lemma_char.split("_MYJOIN_"))
                elif "EMPTY" not in lemma_char:
                    lemma.append(lemma_char)
                #                                        #
                ##########################################
            else:
                lemma.append(lemma_char)

    return inflected_words, lemmata


if __name__ == "__main__":
    # test
    X_train_data, y_train_data = load_dataset("train.dat", data_path="../data", seq2seq=True)