import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt


def plot_loss_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))

    # available flags for customizing: linestyle="--", linewidth=2, marker,
    plt.plot(result['epochs'], result['losses'], label=model_name, marker=".")
    plt.plot(val_result['val_epochs'], val_result['val_losses'], label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(path)


def plot_accuracies_by_model(model_name: str, result: dict, val_result: dict, path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(result['epochs'], result['accuracies'], label=model_name)
    plt.plot(val_result['val_epochs'], val_result['val_accuracies'], label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(path)


def modelPerformance(model_name, optimizer_name,
                     y_train_target, y_train_predicted, y_train_classified,
                     classes, result, val_result):
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
    y_train_target: 1d array-like, or label indicator array / sparse matrix
        Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded
        measure of decisions (as returned by “decision_function” on some classifiers).
    y_train_predicted: 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    y_train_classified: 1d array-like, or label indicator array / sparse matrix
        True binary labels.
    classes: List
        Labels of classifier.
    result: Dictionary
        Loss and accuracy at each epoch during training for each chosen model collected in dictionary.
    val_result: Dictionary
        Loss and accuracy at each epoch during validation for each chosen model collected in dictionary.


    Returns : void
    '''
                         
    acc = accuracy_score(y_train_target, y_train_predicted)
    modelPrecision = precision_score(y_train_target, y_train_predicted)
    modelRecall = recall_score(y_train_target, y_train_predicted)
    f1_measure = f1_score(y_train_target, y_train_predicted)

    cm = confusion_matrix(y_train_target, y_train_predicted)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp/(fp+tn)

    # save metrics in .txt file
    with open('metrics/record_metrics.txt', 'w', encoding='utf8') as w_file:
        w_file.write(f'{model_name}; {optimizer_name} \n')

        w_file.write(('Test accuracy: %.2f %%' % (acc*100)) + '\n')
        w_file.write(('Test Precision: %.2f %%' % (modelPrecision*100)) + '\n')
        w_file.write(('Test Recall or (TPR): %.2f %%' % (modelRecall*100)) + '\n')
        w_file.write(('Test F1-Score: %.2f %%' % (f1_measure*100)) + '\n')
        w_file.write(('Test Fall-out (FPR): %.2f %%' % (fpr*100)) + '\n')
        w_file.write('\n')
        w_file.write('----------')

    # plot roc curve

    fpr, tpr, thresholds = roc_curve(y_train_target, y_train_classified)
    plt.plot(fpr, tpr, linewidth=2, label='')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate, FPR')
    plt.ylabel('True Positive Rate, TPR')
    plt.title('ROC curve')
    plt.savefig(f'metrics/{model_name}_{optimizer_name}_ROC.png')

    # plot precision recall

    precisions, recalls, thresholds = precision_recall_curve(y_train_target, y_train_classified)
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.savefig(f'metrics/{model_name}_{optimizer_name}_Precision_Recall_curve.png')

    # confusion matrices
    e_00, e_11 = cm[0, 0] / (cm[0, 0] + cm[0, 1]), cm[1, 1] / (cm[1, 0] + cm[1, 1])
    weighted_cm = np.array([[e_00, 1 - e_00], [1 - e_11, e_11]])

    cmDisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot()
    plt.savefig(f'metrics/{model_name}_{optimizer_name}_ConfusionMatrix.png')

    w_cmDisp = ConfusionMatrixDisplay(confusion_matrix=weighted_cm, display_labels=classes).plot()
    plt.savefig(f'metrics/{model_name}_{optimizer_name}_WeightedConfusionMatrix.png')

    plot_loss_by_model(model_name, result, val_result, path=f'metrics/{model_name}_{optimizer_name}_losses.png')
    plot_accuracies_by_model(model_name, result, val_result, path=f'metrics/{model_name}_{optimizer_name}_accuracies.png')
