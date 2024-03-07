import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def count_labels(data):
        label_counts = {0: 0, 1: 0}
        for _, sample in enumerate(data):
            if sample != None:
                label_counts[sample['label'][0]] += 1
        return label_counts 


def draw_linechart(df, column_name, fig_name):
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='epoch', y=column_name, hue='legend')
    ax.set_xlabel('epoch')
    ax.set_ylabel('')
    ax.set_title('')
    fig.savefig(f'{fig_name}.jpg')
    

def draw_confusion_matrix(predictions, targets, labels, fig_name):
    # Compute the classification accuracy
    accuracy = (np.array(predictions) == np.array(targets)).mean()
    
    # Create a confusion matrix as a 2D array of zeros
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)
    
    # Update the confusion matrix with the counts
    for target, prediction in zip(targets, predictions):
        confusion_matrix[int(target), int(prediction)] += 1
    
    # Compute the classification accuracy (normalized confusion matrix)
    normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Create a heatmap using Seaborn's heatmap function
    sns.heatmap(normalized_confusion_matrix, annot=True, cmap='Blues', ax=ax, fmt='.2f')
    
    # Set labels, title, and ticks
    ax.set_xlabel('predicted labels')
    ax.set_ylabel('true labels')
    ax.set_title(f'classification accuracy: {accuracy:.2%}')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
      
    # Save plot
    fig.savefig(f'{fig_name}.jpg')