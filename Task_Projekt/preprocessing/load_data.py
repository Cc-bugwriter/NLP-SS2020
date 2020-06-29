def read_label_dataset(file="test-hex06-dataset.txt"):
    """
        read txt data set, adapt for labeled data sets
        :param file: [str], path of local data set (default value： "test-hex06-dataset.txt")
        :param labeled: [boolean], whether the data set is labeled or not
        :return:
        scores: [list], list of scores when labeled
        first_sentences: [list], list of first sentence
        second_sentences: [list], list of second sentence
    """
    # assign path of data sets
    path = "./DATA/"
    with open(f"{path}{file}", 'r', encoding='utf-8') as file:
        # labeled data
        scores = []
        first_sentences = []
        second_sentences = []
        for line in file.readlines():
            score, first_sentence, second_sentence = line.strip().split('\t')
            scores.append(float(score))
            first_sentences.append(first_sentence)
            second_sentences.append(second_sentence)

        # string to list
        first_sentences = [sentence.split() for sentence in first_sentences]
        second_sentences = [sentence.split() for sentence in second_sentences]

        return scores, first_sentences, second_sentences


def read_unlabel_dataset(file: str):
    """
        read txt data set, adapt for unlabeled data sets
        :param file: [str], path of local data set
        :return:
        first_sentences: [list], list of first sentence
        second_sentences: [list], list of second sentence
    """
    # assign path of data sets
    path = "./DATA/"
    with open(f"{path}{file}", 'r', encoding='utf-8') as file:
        # unlabeled data
        first_sentences = []
        second_sentences = []
        for line in file.readlines():
            first_sentence, second_sentence = line.strip().split('\t')
            first_sentences.append(first_sentence)
            second_sentences.append(second_sentence)

        # string to list
        first_sentences = [sentence.split() for sentence in first_sentences]
        second_sentences = [sentence.split() for sentence in second_sentences]

        return first_sentences, second_sentences
