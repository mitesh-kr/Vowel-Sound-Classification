# -*- coding: utf-8 -*-
"""
Classification module for vowel classification system
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def prepare_data(features_df):
    """
    Prepare data for classification by standardizing and applying PCA.
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing formant features
    
    Returns:
    --------
    X_train : ndarray
        Training features
    X_test : ndarray
        Testing features
    y_train : ndarray
        Training labels
    y_test : ndarray
        Testing labels
    """
    # Standardize features BEFORE PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df[['F0', 'F1', 'F2', 'F3']])

    # Apply PCA after standardization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Print explained variance ratio
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, features_df['vowel'], test_size=0.2, random_state=45
    )
    
    return X_train, X_test, y_train, y_test

def train_knn_classifier(X_train, y_train, n_neighbors=11):
    """
    Train a K-Nearest Neighbors classifier.
    
    Parameters:
    -----------
    X_train : ndarray
        Training features
    y_train : ndarray
        Training labels
    n_neighbors : int, optional
        Number of neighbors, default is 11
    
    Returns:
    --------
    knn : KNeighborsClassifier
        Trained KNN classifier
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_classifier(classifier, X_test, y_test):
    """
    Evaluate the classifier performance.
    
    Parameters:
    -----------
    classifier : object
        Trained classifier
    X_test : ndarray
        Testing features
    y_test : ndarray
        Testing labels
    
    Returns:
    --------
    accuracy : float
        Classification accuracy
    y_pred : ndarray
        Predicted labels
    """
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'Accuracy: {accuracy:.2f}%')
    print('Classification Report:\n', classification_report(y_test, y_pred))
    return accuracy, y_pred
