import matplotlib.pyplot as plt
import numpy as np

def compare_gt_pred_histogram(gt_labels, pred_labels_r, semantic3d_labels):
    # Plot histograms of ground truth and predicted labels
    plt.figure(figsize=(10, 5))
    bins = np.arange(len(semantic3d_labels) + 1) - 0.5
    plt.hist(gt_labels, bins=bins, alpha=0.5, label='Ground Truth')
    plt.hist(pred_labels_r, bins=bins, alpha=0.5, label='Prediction')
    plt.xticks(ticks=range(len(semantic3d_labels)), labels=[semantic3d_labels[i] for i in range(len(semantic3d_labels))], rotation=45)
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.title('Histogram of Ground Truth vs Prediction')
    plt.legend()
    plt.tight_layout()
    plt.show()