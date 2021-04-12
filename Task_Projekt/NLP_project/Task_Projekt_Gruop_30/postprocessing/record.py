import pickle


def score_writer(file: str, scores: [float]):
    """
        Writes the given list of scores to the file with the given filename.
        :param file: [str], path of local data set (sentences)
        :param scores: [list], scores of labeled sentences
    """
    with open(file, mode='w', encoding='utf8') as file:
        file.write('\n'.join(str(score) for score in scores))


def save_params(name: str, data):
    """
        save Intermediate variables in local files
        :param name: [str], path of local data set (sentences)
        :param data: [array], saved data
    """
    data_save = open(f'./result/{name}.pkl', 'wb')
    pickle.dump(data, data_save)
    data_save.close()


def load_params(name: str):
    """
        load Intermediate variables from local files
        :param name: [str], path of local data set (sentences)
        :param data: [array], saved data
    """
    data_load = open(f'./result/{name}.pkl', 'rb')
    data = pickle.load(data_load)
    data_load.close()

    return data
