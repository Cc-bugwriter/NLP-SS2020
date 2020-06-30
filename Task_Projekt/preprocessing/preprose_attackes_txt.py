import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def get_unattacked_sentences(sentences):
    """
    xxxx
    """
    # load the unicode descriptions into a single dataframe with the chars as indices
    descs = pd.read_csv('./DATA/NamesList.txt', skiprows=np.arange(16), header=None, names=['code', 'description'],
                        delimiter='\t')
    descs = descs.dropna(0)
    descs_arr = descs.values  # remove the rows after the descriptions
    vectorizer = CountVectorizer(max_features=1000)
    desc_vecs = vectorizer.fit_transform(descs_arr[:, 0]).astype(float)
    vec_colnames = np.arange(desc_vecs.shape[1])
    desc_vecs = pd.DataFrame(desc_vecs.todense(), index=descs.index, columns=vec_colnames)
    descs = pd.concat([descs, desc_vecs], axis=1)

    def get_unattacked_char(ch):
        """
        xxxx
        """
        def char_to_hex_string(ch):
            """
            xxxx
            """
            # function for retrieving the variations of a character
            return '{:04x}'.format(ord(ch)).upper()

        # get unicode number for c
        c = char_to_hex_string(ch)

        # problem: latin small characters seem to be missing?
        if np.any(descs['code'] == c):
            description = descs['description'][descs['code'] == c].values[0]
        # else:
        #     print('Failed to disturb %s, with code %s' % (ch, c))
        #     return c, np.array([])

        # strip away everything that is generic wording, e.g. all words with > 1 character in
        toks = description.split(' ')
        case = 'unknown'
        identifiers = []
        for tok in toks:
            if len(tok) == 1:
                identifiers.append(tok)
                # for debugging
                if len(identifiers) > 1:
                    case == "unknown"
            elif tok == 'SMALL':
                case = 'SMALL'
            elif tok == 'CAPITAL':
                case = 'CAPITAL'
        if len(identifiers) == 0:
            return ch
        if case == "unknown":
            return ch
        elif case == "CAPITAL":
            return identifiers[0].upper()

        return identifiers[0].lower()

    unattacked_sentences = list()
    for sentence in sentences:
        ccc = [''.join([get_unattacked_char(ch) for ch in word]) for word in sentence]
        unattacked_sentences.append(ccc)

    return unattacked_sentences







