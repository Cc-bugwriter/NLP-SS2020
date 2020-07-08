import os
from model import bilstm, seq2seq


def run(params):
    # set model output path
    params["model_path"] = os.path.join("results", "{}.model".format(params["model"]))

    # create model
    if params["model"] == "bilstm":
        model = bilstm.Hex10BiLSTM(params)
    elif params["model"] == "seq2seq":
        model = seq2seq.Hex10Seq2Seq(params)
    else:
        raise ValueError("Unknown model: {}".format(params["model"]))

    # get predictions: list of tuples [(truth_1, pred_1), (truth_2, pred_2), ...]
    predictions = model.train_and_predict()

    # write predictions to file
    outfile = os.path.join("results", "{}_predictions.txt".format(params["model"]))
    with open(outfile, 'w') as file:
        file.write("\t\t".join(["input", "prediction", "truth"]) + "\n")
        file.write("-"*35+"\n")
        for line in predictions:
            file.write("\t\t".join(line) + "\n")

    # print word accuracy
    tp = sum([1 if t == p else 0 for _,p,t in predictions])
    acc = tp / len(predictions)
    print("Word accuracy of {}: {}".format(params["model"], acc))


if __name__=='__main__':
    params = {"model": "seq2seq",    # bilstm or seq2seq
              "batch_size": 40,
              "dropout": 0.5,
              "hidden_units": 60,
              "epochs": 10}
    run(params)