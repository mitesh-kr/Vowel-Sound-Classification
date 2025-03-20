# -*- coding: utf-8 -*-
"""
Main script for vowel classification system
"""

import os
import argparse
from feature_extraction import load_and_extract_features
from visualization import (
    plot_f1_f2_vowel_space, 
    plot_formant_distributions, 
    plot_f0_distribution,
    plot_confusion_matrix
)
from classification import prepare_data, train_knn_classifier, evaluate_classifier

def main(dataset_path):
    """
    Main function to run the vowel classification system.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset directory
    
    Returns:
    --------
    None
    """
    print("Step 1: Extracting features...")
    features_df = load_and_extract_features(dataset_path)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    print("\nStep 2: Visualizing features...")
    plot_f1_f2_vowel_space(features_df)
    plot_formant_distributions(features_df)
    plot_f0_distribution(features_df)
    
    print("\nStep 3: Preparing data for classification...")
    X_train, X_test, y_train, y_test = prepare_data(features_df)
    
    print("\nStep 4: Training KNN classifier...")
    knn = train_knn_classifier(X_train, y_train)
    
    print("\nStep 5: Evaluating classifier...")
    accuracy, y_pred = evaluate_classifier(knn, X_test, y_test)
    
    print("\nStep 6: Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, title='KNN Confusion Matrix')
    
    print("\nVowel classification completed successfully!")
    print(f"Results saved in the 'results' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vowel Classification System")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to the vowel dataset directory"
    )
    args = parser.parse_args()
    
    main(args.dataset_path)
