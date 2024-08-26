import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, fbeta_score, roc_auc_score, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import pandas as pd

from config import settings

def plot_loss_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))

    # available flags for customizing: linestyle="--", linewidth=2, marker,
    plt.plot(result['epochs'], result['losses'], label="train", marker=".")
    plt.plot(val_result['val_epochs'], val_result['val_losses'], label="valid", marker="*")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(path)
    plt.close()



def plot_accuracies_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(result['epochs'], result['accuracies'], label="train", marker=".")
    plt.plot(val_result['val_epochs'], val_result['val_accuracies'], label="valid", marker="*")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(path)
    plt.close()

def modelPerformance(model_name, optimizer_name,
                     y_target, y_predicted, y_classified,
                     classes, 
                     f_beta = 2,
                    #  result, val_result
                     ):
    '''
    Plots ROC and Precision-Recall curves, change of loss and accuracy throughout training,
    confusion matrix and its weighted version and saves them in .png files,
    counts accuracy, precision, recall, false positive rate and f1-score and saves them in .txt file

    Parameters
    ----------
    model_name: string
        Name of model.
    optimizer_name: string
        Name of optimizer.
    y_target: 1d array-like, or label indicator array / sparse matrix
        Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded
        measure of decisions (as returned by “decision_function” on some classifiers).
    y_predicted: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    y_classified: 1d array-like, or label indicator array / sparse matrix
        True binary labels.
    classes: List
        Labels of classifier.
    result: Dictionary
        Loss and accuracy at each epoch during training for each chosen model collected in dictionary.
    val_result: Dictionary
        Loss and accuracy at each epoch during validation for each chosen model collected in dictionary.


    Returns : void
    '''
                         
    acc = accuracy_score(y_target, y_predicted)
    precision = precision_score(y_target, y_predicted)
    recall = recall_score(y_target, y_predicted)
    f1_measure = f1_score(y_target, y_predicted)
    fbeta_measure = fbeta_score(y_target, y_predicted, beta=f_beta)

    cm = confusion_matrix(y_target, y_predicted)
    tn, fp, fn, tp = cm.ravel()
    fpr_measure = fp/(fp+tn)

    roc_auc = roc_auc_score(y_target, y_predicted)

    model_path = Path(settings.METRICS_PATH, f"{model_name}_{optimizer_name}")
    os.makedirs(model_path, exist_ok=True)

    # plot roc curve

    fpr, tpr, _ = roc_curve(y_target, y_classified)
    plt.plot(fpr, tpr, linewidth=2, label='')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate, FPR')
    plt.ylabel('True Positive Rate, TPR')
    plt.title('ROC curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(Path(model_path, 'roc_curve.png'))
    plt.close()
    # plot precision recall

    precisions, recalls, _ = precision_recall_curve(y_target, y_classified)
# Step 6: Calculate Area Under the PR curve.
    pr_auc = auc(recalls, precisions)
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(Path(model_path, 'precision_recall_curve.png'))
    plt.close()

    # confusion matrices
    e_00, e_11 = cm[0, 0] / (cm[0, 0] + cm[0, 1]), cm[1, 1] / (cm[1, 0] + cm[1, 1])
    weighted_cm = np.array([[e_00, 1 - e_00], [1 - e_11, e_11]])

    _ = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    plt.savefig(Path(model_path, 'confusion_matrix.png'))
    plt.close()

    _ = ConfusionMatrixDisplay(confusion_matrix=weighted_cm, display_labels=classes).plot()
    plt.savefig(Path(model_path, 'weighted_confusion_matrix.png'))
    plt.close()

    metrics = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall (TPR)": recall,
        "Fall-out (FPR)": fpr_measure,
        "PR AUC": pr_auc,
        "ROC AUC": roc_auc,
        "F-1 score": f1_measure,
        "Beta": f_beta,
        "F-beta score": fbeta_measure,
    }

    with open(Path(model_path, 'metrics.json'), "w") as file:
        json.dump(metrics, file)

def combine_metrics(selected_models: list, optimizer_name):
    for model_name, _ in selected_models:

        all_metrics = {}
        for model_name, _ in selected_models:
            combination = f"{model_name}_{optimizer_name}"
            metrics_path = Path(settings.METRICS_PATH, combination, "metrics.json")

            with open(metrics_path) as file:
                all_metrics[combination] = json.load(file)

        metrics_frame = pd.DataFrame(all_metrics).T
        metrics_frame.index.name = "Combination"

        metrics_frame.to_csv(Path(settings.METRICS_PATH, "metrics.csv"))

    return metrics_frame