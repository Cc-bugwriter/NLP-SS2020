def score_writer(file: str, scores: [float]):
    """
        Writes the given list of scores to the file with the given filename.
        :param file: [str], path of local data set (sentences)
        :param scores: [list], scores of labeled sentences
    """
    with open(file, mode='w', encoding='utf8') as file:
        file.write('\n'.join(str(score) for score in scores))