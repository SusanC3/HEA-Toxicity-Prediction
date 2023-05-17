import Data
import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt
from data_assembly import Normalizer

dataset = Data.Dataset()

device = torch.device("cuda:0")

def score_model(model, val_idx):
    X_test, y_test = [], []
    for id in val_idx:
        X_test.append(dataset.__getitem__(id)[0])
        y_test.append(dataset.__getitem__(id)[1]) 

    #pdb.set_trace()

    X_test = torch.from_numpy(np.array(X_test)).to(device)
    #y_test = torch.from_numpy(np.array(y_test)).to(device)

    # output = dataset.output_normalizer.decode(model(X_test)).cpu().data.numpy()
    # real = dataset.output_normalizer.decode(y_test).cpu().data.numpy()
    output = model(X_test).cpu().data.numpy()
    real = np.array(y_test)

    plt.title("Activity score predictions of all compounds")
    plt.hist(output, bins=100)
    plt.xlabel("Prediction value")
    plt.ylabel("Occurance")
    plt.savefig("outputs")
    plt.clf()

    plt.title("Actual activity score of all compounds")
    plt.hist(real, bins=100)
    plt.xlabel("Value")
    plt.ylabel("Occurance")
    plt.savefig("real")
    plt.clf()

    pdb.set_trace()

    # output = model(X_test).cpu().detach().numpy()
    # real = y_test.cpu().numpy()
    percent_errors = []
    zero_guesses = []
    for i in range(len(output)):
        if real[i] != 0.0:
            percent_errors.append( abs(output[i] - real[i]) / real[i] * 100 )
        else:
            zero_guesses.append(output[i])


    percent_errors = np.array(percent_errors)
    zero_guesses = np.array(zero_guesses)

    perc_mean = np.mean(percent_errors)
    perc_std = np.std(percent_errors)
    zero_mean = np.mean(zero_guesses)
    zero_std = np.std(zero_guesses)

    print("perc mean:", str(perc_mean), "perc std:", perc_std)
    print("zero mean:", str(zero_mean), "zero std:", zero_std)
    print()

  #  return [perc_mean, perc_std, zero_mean, zero_std]
