import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names=None, output_path=None):
    """
    Compact confusion matrix plot with larger numbers - text only on left side
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Create a more compact figure
    fig_size = max(4, len(class_names) * 0.8)  # Dynamic sizing based on number of classes
    plt.figure(figsize=(fig_size, fig_size * 0.9))
    
    # Create heatmap with larger annotations
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14, 'weight': 'bold'},  # Bigger, bolder numbers
                    cbar=False,  # Remove colorbar to save space
                    square=True)  # Make cells square for better proportions
    
    # Adjust font sizes for labels
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Remove bottom tick labels
    plt.xticks([])  # Remove x-axis tick labels (bottom)
    plt.yticks(fontsize=10, rotation=0)  # Keep y-axis labels horizontal
    
    # Move the "True" label to the right side
    ax.yaxis.set_label_position("right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()
    
    # Print summary stats
    total = np.sum(cm)
    accuracy = np.trace(cm) / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")