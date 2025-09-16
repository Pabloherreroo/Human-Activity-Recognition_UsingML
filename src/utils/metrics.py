import numpy as np
from .plot_metrics import plot_confusion_matrix

def get_accuracy(confusion_matrix):
    """Calculates the accuracy from a confusion matrix."""
    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def check_best(pipeline, confusion_matrix, model_path):
    """Checks if its the best performing model and saves"""
    if model_path:
        cm_path = f"results/{model_path.split('/')[-1].replace('.joblib', '.png')}"
        plot_confusion_matrix(confusion_matrix, pipeline.labels, output_path=cm_path)
    else:
        plot_confusion_matrix(confusion_matrix, pipeline.labels)
