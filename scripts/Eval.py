from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

# Класс для метрик и матрицы неточностей
class EvalClassifier:
    def __init__(self, model):
        self.model = model

    def get_confusion_matrix(self, y_true, y_pred, classes):
        class_idx = {label: index for index, label in enumerate(classes)}
        cm = np.zeros((len(classes), len(classes)), dtype=int)
        for true_label, pred_label in zip(y_true, y_pred):
            cm[class_idx[true_label]][class_idx[pred_label]] += 1
        return cm

    def calculate_metrics_by_class(self, confusion_matrix, classes):
        metrics_table = []
        for i in range(len(classes)):
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            tn = np.sum(confusion_matrix) - (tp + fp + fn)
            accuracy = (tp + tn) / (tp + fp + fn + tn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics_table.append([accuracy, precision, recall, specificity, f1_score])

        metrics_df = pd.DataFrame(metrics_table, index=classes,
                                  columns=["accuracy", "precision", "recall", "specificity", "f1-score"])
        return metrics_df

    def plot_metrics_by_class(self, metrics, title):
        plt.figure(figsize=(9, 7))
        sbn.heatmap(metrics, annot=True, fmt=".4f", cmap="Blues", cbar=True)
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.title(title)
        plt.show()

    def evaluate_test(self, y_test, y_pred_test, classes):
        cm_test = self.get_confusion_matrix(y_test, y_pred_test, classes)
        self.plot_confusion_matrix(cm_test, classes)
        metrics_test = self.calculate_metrics_by_class(cm_test, classes)
        self.plot_metrics_by_class(metrics_test, "Classification metrics for the test set")
        return metrics_test

    def evaluate_train(self, y_train, y_pred_train, classes):
        cm_train = self.get_confusion_matrix(y_train, y_pred_train, classes)
        self.plot_confusion_matrix(cm_train, classes)
        metrics_train = self.calculate_metrics_by_class(cm_train, classes)
        self.plot_metrics_by_class(metrics_train, "Classification metrics for the training set")
        return metrics_train

    def plot_confusion_matrix(self, cm, classes):
        plt.figure(figsize=(8, 6))
        sbn.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True marks")
        plt.title("Confusion matrix")
        plt.show()

