"""
Library functions to compute different metrics for tasks.
"""

__author__ = "Ashwinkumar Ganesan"
__email__ = "gashwin1@umbc.edu"

from tabulate import tabulate
import math
import matplotlib
import matplotlib.pyplot as plt
import torch
import xclib.evaluation.xc_metrics as xc_metrics
import wandb
import numpy as np

# Compute the precision score for multi-label binary classification task.
def mbprecision(y_true, y_pred):
    correct_pred = torch.sum(y_pred & y_true, axis=1).float()
    print(correct_pred.dtype)
    return torch.mean(correct_pred / torch.sum(y_true, axis=1))

# Compute the recall score for multi-label binary classification task.
def mbrecall(y_true, y_pred):
    return torch.mean(torch.sum(y_pred & y_true, axis=1) / torch.sum(y_true, axis=1))


def plot_tr_stats(tr_stats, th_stats, spoch, sth, filename):
    """
    Plot stats about the experiment.
    tr_stats: Training statistics (includes loss, precision, recall and F1)
    th_stats: Grid search statistics for configuring threshold.
    epochs: Number of epochs that the model is trained for.
    spoch: epoch that has optimal paramaters.
    sth: optimal threshold.
    filename: location to store plots.
    """
    fig, ax = plt.subplots(3, figsize=(10, 10))

    ep = tr_stats['Epoch']
    tr_loss = tr_stats['Training Loss']
    val_loss = tr_stats['Val Loss']
    pr = tr_stats['Precision']
    re = tr_stats['Recall']
    f1 = tr_stats['F1 Score']
    th = th_stats['Threshold']

    ax[0].plot(ep, tr_loss)
    ax[0].plot(ep, val_loss)
    ax[0].set_title("Training & Validation Loss Per Epoch", size=16)
    ax[0].set_xlabel("Epoch", size=14)
    ax[0].set_ylabel("Loss", size=14)
    ax[0].legend(["Training Loss", "Validation Loss"], fontsize="large")
    ax[0].axvline(x=spoch, linestyle='dashed')

    ax[1].plot(ep, pr)
    ax[1].plot(ep, re)
    ax[1].plot(ep, f1)
    ax[1].set_title("Validation Precision, Recall & F-1 Score \n (Threshold = 0.25)", size=16)
    ax[1].set_xlabel("Epoch", size=14)
    ax[1].set_ylabel("Score", size=14)
    ax[1].legend(["Validation Precision", "Validation Recall", "Validation F1 Score"], fontsize="large")
    ax[1].axvline(x=spoch, linestyle='dashed')

    ax[2].plot(th, th_stats['Precision'])
    ax[2].plot(th, th_stats['Recall'])
    ax[2].plot(th, th_stats['F1 Score'])
    ax[2].set_title("Validation Precision, Recall & F-1 Score \n Optimize Threshold", size=16)
    ax[2].set_xlabel("Theshold", size=14)
    ax[2].set_ylabel("Score", size=14)
    ax[2].legend(["Validation Precision", "Validation Recall", "Validation F1 Score"], fontsize="large")
    ax[2].axvline(x=sth, linestyle='dashed')

    fig.tight_layout()
    plt.savefig(filename + ".png")

# Adapted from: https://github.com/kunaldahiya/pyxclib
def compute_inv_propensity(train_labels, A=0.55, B=1.5):
    """
        Compute Inverse propensity values
        Values for A/B:
            Wikpedia-500K: 0.5/0.4
            Amazon-670K, Amazon-3M: 0.6/2.6
            Others: 0.55/1.5

        Arguments:
        train_labels : numpy ndarray
    """
    inv_propen = xc_metrics.compute_inv_propesity(train_labels, A, B)
    return inv_propen

def compute_Nl_dict(train_labels):
    """
    Compute the occurrence count of each label in the training dataset.

    Arguments:
    train_labels : numpy ndarray
        2D array where each row represents a data point and each column represents a label.
        Each element is either 0 or 1, indicating whether the label is present or not.

    Returns:
    Nl_dict : dict
        Dictionary mapping each label to its occurrence count.
    """
    num_labels = train_labels.shape[1]  # Assuming labels are in columns
    Nl_dict = {l: np.sum(train_labels[:, l]) for l in range(num_labels)}

    return Nl_dict

# Compute metrics with propensity.
def compute_prop_metrics(true_labels, predicted_labels, inv_prop_scores, topk=10):
    """Compute propensity weighted precision@k and DCG@k.
       Arguments:
       true_labels : numpy ndarray
                     Ground truth labels from the dataset (one-hot vector).
       predicted_labels : numpy ndarray
                          Predicted labels (one-hot vector of labels)
    """
    acc = xc_metrics.Metrics(true_labels=true_labels, inv_psp=inv_prop_scores,
                             remove_invalid=False)
    return acc.eval(predicted_labels, topk)


# Print the final results.
# This provides the results for agg metrics when threshold for inference
# is optimized and metrics are then computed.
def display_agg_results(args, te_loss, pr, rec, f1):
    print("----------Tests with Threshold Inference------------")
    print("Inference Threshold: {:.3f}".format(args.th))
    print("Test Loss: {:.3f}".format(te_loss))
    print("Test Precision: {:.3f}".format(pr * 100))
    print("Test Recall: {:.3f}".format(rec * 100))
    print("Test F1-Score: {:.3f}\n".format(f1 * 100))





def display_metrics(metrics,spn_dim,h_size, dataset,k=10):

    # Merge batchwise metrics.
    final_metrics = [[0.0] * k,[0.0] * k,[0.0] * k,[0.0] * k]

    for idx, metric in enumerate(metrics):
        for i in range(0, 4):
            for j in range(0, k):
                final_metrics[i][j] += metric[i][j]
    # total_mrr = metrics[-1] 
    # if isinstance(total_mrr, list):
    #     total_mrr = total_mrr[0] 
    

    # Dataset metrics.
    print("----------Tests with Ordered Retrieval------------")
    table = [['Precision@k'] + [i * 100 / (idx + 1) for i in final_metrics[0]]]
    table.append(['nDCG@k'] + [i * 100 / (idx + 1) for i in final_metrics[1]])
    table.append(['PSprec@k'] + [i * 100 / (idx + 1) for i in final_metrics[2]])
    table.append(['PSnDCG@k'] + [i * 100 / (idx + 1) for i in final_metrics[3]])
    print(tabulate(table, headers=[i+1 for i in range(0, k)],
                   floatfmt=".3f"))
    
    wandb.log({"P@1": table[0][1],"P@3": table[0][3],"P@5": table[0][5], "P@10": table[0][10], "P@20": table[0][20],"P@50": table[0][50],
            "nDCG@1": table[1][1],"nDCG@3": table[1][3],"nDCG@5": table[1][5], "nDCG@10": table[1][10], "nDCG@20": table[1][20],"nDCG@50": table[1][50],
            "PSP@1": table[2][1], "PSP@3": table[2][3],"PSP@5": table[2][5], "PSP@10": table[2][10], "PSP@20": table[2][20],"PSP@50": table[2][50], "Dataset": dataset,
            },step=spn_dim)
