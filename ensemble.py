import pandas as pd
import numpy as np


def LabelSmoothing(encodings , alpha):
    K = encodings.shape[1]
    y_ls = (1 - alpha) * encodings + alpha / K
    return y_ls


def main():
    results = [pd.read_csv('submission_effnet_ns.csv'),         ## Noisy Student
                pd.read_csv('submission_densenet.csv'),         ## DenseNet
                pd.read_csv('submission_effnet.csv')]           ## EffNet

    n = len(results)

    ## Do ensembling
    results_np = np.zeros((n, 1821, 4))
    for i,df in enumerate(results):
        temp = df.to_numpy()
        results_np[i] = (temp[:,1:])
    results_ensembled = np.sum(results_np, axis=0) / n

    ## Apply label smoothing
    smoothed_results = LabelSmoothing(results_ensembled, 0.01)

    ## Save File
    test = pd.read_csv('test.csv')
    sub = pd.DataFrame(smoothed_results, columns=['healthy','multiple_diseases','rust','scab'])
    sub['image_id'] = test.image_id
    sub = sub[['image_id','healthy','multiple_diseases','rust','scab']]
    sub.to_csv("submission_ensemble.csv", index = False)


if __name__== "__main__":
    main()