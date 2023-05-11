import Data
import torch
import numpy as np
import pdb

dataset = Data.Dataset()

device = torch.device("cuda:0")

def score_model(model, val_idx):
    X_test, y_test = [], []
    for id in val_idx:
        X_test.append(dataset.__getitem__(id)[0])
        y_test.append(dataset.__getitem__(id)[1]) 

    X_test = torch.from_numpy(np.array(X_test)).to(device)
    y_test = torch.from_numpy(np.array(y_test)).to(device)

    output = dataset.output_normalizer.decode(model(X_test)).cpu().data.numpy()
    real = dataset.output_normalizer.decode(y_test).cpu().data.numpy()
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
