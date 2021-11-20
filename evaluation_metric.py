from __future__ import print_function, division
import numpy as np


def rmse(predictions, targets):
    return np.sqrt(((predictions.numpy() - targets.numpy()) ** 2).mean())


def mae(predictions, targets):
    return np.mean(np.abs(predictions.numpy() - targets.numpy()))


def PCC(predictions, targets):
    # torch:[bs,size,len]
    B, C, T = predictions.shape

    def singleBatchPCC(x, y):
        # numpy:[size,len]
        pcc = np.abs(np.diagonal(np.corrcoef(x, y)[0:C, C:])).mean()
        return pcc

    predictions = np.array(predictions)
    targets = np.array(targets)
    pcc_value = np.mean([singleBatchPCC(predictions[i], targets[i]) for i in range(B)])
    return pcc_value


def CCC(predictions, targets, sample_weight=None, multioutput='uniform_average'):
    # Input torch:[bs,size,len]
    B, C, T = predictions.shape
    def singleBatchCCC(x,y):
        # Input numpy:[size,len]
        cor = np.abs(np.diagonal(np.corrcoef(x, y)[0:C, C:]))
        mean_true = np.mean(y,axis=-1)
        mean_pred = np.mean(x,axis=-1)

        var_true = np.var(y,axis=-1)
        var_pred = np.var(x,axis=-1)

        sd_true = np.std(y,axis=-1)
        sd_pred = np.std(x,axis=-1)

        numerator = 2 * cor * sd_true * sd_pred

        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        return (numerator / denominator).mean()

    predictions = np.array(predictions)
    targets = np.array(targets)
    ccc_value = np.mean([singleBatchCCC(predictions[i], targets[i]) for i in range(B)])
    return ccc_value
