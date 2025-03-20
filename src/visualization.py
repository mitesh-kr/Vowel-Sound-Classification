# -*- coding: utf-8 -*-
"""
Visualization module for vowel classification system
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_f1_f2_vowel_space(features_df):
    """
    Plot the vowel space using F1 vs F2 as scatter plot.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing formant features
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=features_df, x='F2', y='F1', hue='vowel', style='category', palette='tab10', alpha=0.7
    )
    plt.xlabel('F2 Frequency (Hz)')
    plt.ylabel('F1 Frequency (Hz)')
    plt.title('Vowel Space (F1 vs F2)')
    plt.gca().invert_yaxis()  # Invert y-axis to match traditional vowel space plots
    plt.legend(title='Vowel')
    plt.savefig('results/vowel_space.png')
    plt.show()

def plot_formant_distributions(features_df):
    """
    Plot box plots for F1, F2, and F3 grouped by vowel.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing formant features
    
    Returns:
    --------
    None
    """
    formant_features = ['F1', 'F2', 'F3']

    for formant in formant_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=features_df, x='vowel', y=formant, palette='tab10')
        plt.title(f'Box Plot of {formant} by Vowel')
        plt.xlabel('Vowel')
        plt.ylabel(f'{formant} Frequency (Hz)')
        plt.savefig(f'results/{formant}_distribution.png')
        plt.show()

def plot_f0_distribution(features_df):
    """
    Plot box plot for F0 grouped by vowel.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing formant features
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=features_df, x='vowel', y='F0', palette='coolwarm')
    plt.title('Box Plot of F0 by Vowel')
    plt.xlabel('Vowel')
    plt.ylabel('F0 Frequency (Hz)')
    plt.savefig('results/F0_distribution.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot confusion matrix for classification results.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    title : str, optional
        Title for the plot, default is 'Confusion Matrix'
    
    Returns:
    --------
    None
    """
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.png')
    plt.show()
