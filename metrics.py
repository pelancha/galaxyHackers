import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt


def plot_roc_curve(y_train_target, y_train_classified, label=None): #y_train_classified = {-1, 1} - result was classified as positive or negative
    fpr, tpr, thresholds = roc_curve(y_train_target, y_train_classified)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate, FPR')
    plt.ylabel('True Positive Rate, TPR')
    plt.title('ROC curve')
    plt.savefig("ROC.png")
    plt.show()


def plot_precision_recall(y_train_target, y_train_classified):
    precisions, recalls, thresholds = precision_recall_curve(y_train_target, y_train_classified)
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.savefig("Precision_Recall_curve.png")
    plt.show()


# TODO: in model structure
# - Create lists like train_accuracies, train_losses, validation_accuracies, validation_losses 
# - Save accuracies and losses after each epoch in these lists


# TODO: when you switch between training and validation stages via "for"
# Modify function s.t. 
# - input is 1. y_train_target 2. current y_train_predicted (after each one epoch) 3. list of accuracies 4. list of losses
# - output is 1. updated list of accuracies 2. updated list of losses


# TODO(maybe):
# create class of model that will have fields for all losses and accuracies
# GOAL(maybe): access to plotProgress from modelPerfomance


def plotProgress(train_accuracy, val_accuracy, train_loss, val_loss):
    epochs = range(len(train_accuracy))
# plot training vs validation accuracy change
    plt.plot(epochs, train_accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
# plot training vs validation loss change
    plt.figure()
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def modelPerfomance(model, mHistory, y_train_target, y_train_predicted, y_train_classified, classes): #parametrs: tested model, fitted estimator (result of model.fit()), classes - labels of classifier
    acc = accuracy_score(y_train_target, y_train_predicted)
    modelPrecision = precision_score(y_train_target, y_train_predicted)
    modelRecall = recall_score(y_train_target, y_train_predicted)
    f1_measure = f1_score(y_train_target, y_train_predicted)

    cm = confusion_matrix(y_train_target, y_train_predicted)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp/(fp+tn)

    print('Test accuracy: %.2f %%' % (acc*100))
    print('Test Precision: %.2f %%' % (modelPrecision*100))
    print('Test Recall or (TPR): %.2f %%' % (modelRecall*100))
    print('Test F1-Score: %.2f %%' % (f1_measure*100))
    print('Test Fall-out (FPR): %.2f %%' % (fpr*100))

    cmDisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot() 
    plt.show()

    plot_roc_curve(y_train_target, y_train_classified)
    plot_precision_recall(y_train_target, y_train_classified)
