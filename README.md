# Vowel Classification System

This repository contains a Python-based system for classifying five different vowel sounds (/a/, /e/, /i/, /o/, /u/) using traditional speech processing techniques.


## [Google colab link](https://colab.research.google.com/drive/1yxLwj0Z4YthIge3AQh6exPbVY5gra1b3?usp=sharing)

## Overview

The system uses formant frequencies (F1, F2, F3) and fundamental frequency (F0) extracted from speech samples to classify vowels. It employs Linear Predictive Coding (LPC) for formant extraction and an improved autocorrelation method for fundamental frequency estimation.

### Key Features:

- **Feature Extraction**: Extracts formant frequencies (F1, F2, F3) and fundamental frequency (F0) from audio samples
- **Vowel Space Visualization**: Plots the vowel space using F1-F2 plots
- **Classification**: Implements K-Nearest Neighbors classification algorithm
- **Performance Evaluation**: Provides confusion matrices and accuracy metrics

## Repository Structure

```
vowel_classification/
├── README.md               # Project documentation
├── LICENSE                 # MIT License
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore file
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── preprocess.py       # Audio preprocessing functions
│   ├── feature_extraction.py # Feature extraction code
│   ├── visualization.py    # Visualization functions
│   ├── classification.py   # Classification algorithms
│   └── main.py             # Main script
└── results/                # Output directory
    └── .gitkeep            # Placeholder to include empty dir
```

## Dataset

This project uses the "Vowel database: adults" dataset, which includes recordings from adult males and adult females. The dataset is divided into training and testing sets with an 80:20 ratio and a random state of 45.
[Download Link](https://drive.google.com/drive/folders/1q1_bgzXWSS-cTy646rAa0TfvL_w0K70_)
## Installation

### Clone the Repository

```bash
git clone https://github.com/mitesh-kr/Vowel-Sound-Classification.git
cd Vowel-Sound-Classification
```

### Install Dependencies

To install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the "Vowel database: adults" dataset
2. Update the `dataset_path` in `main.py` to point to your dataset location
3. Run the main script:

```bash
python src/main.py --dataset_path="/path/to/your/dataset"
```

## Components

### 1. Feature Extraction (src/feature_extraction.py)

- Implements Linear Predictive Coding (LPC) to extract formant frequencies
- Extracts fundamental frequency using autocorrelation
- Processes audio files with pre-emphasis, framing, and windowing

### 2. Visualization (src/visualization.py)

- Creates F1-F2 vowel space plots
- Generates formant distribution visualizations
- Produces F0 distribution plots by vowel

### 3. Classification (src/classification.py)

- Implements K-Nearest Neighbors classifier
- Applies Principal Component Analysis (PCA) for dimensionality reduction
- Generates confusion matrices and classification reports

## Results

The system achieves good classification accuracy by leveraging the discriminative power of formant frequencies. The F1-F2 vowel space clearly shows the separation between different vowels, which aligns with theoretical expectations.

For training, the KNN algorithm has been used to improve the accuracy of PCA, which
has been done for the extracted feature values of F1, F2, F3 and F0.
The optimal accuracy of 58.33 % was achieved for PCA components equal to 3 and
n_neighbors=11 for the KNN algorithm.


## Analysis and Reflection

### Theoretical Alignment

The extracted formant patterns align well with theoretical expectations:
- /a/ has high F1 and low F2
- /i/ has low F1 and high F2
- /u/ has low F1 and low F2
- /e/ and /o/ occupy intermediate positions

### Potential Sources of Error

- Speaker variability (age, accent, speech rate)
- Background noise or recording quality issues
- Overlapping formant regions between similar vowels (e.g., /e/ and /i/)

### Historical Context

This approach relates to historical speech recognition systems through:
- Use of formant-based features, which were central to early speech recognition
- Application of LPC, a technique developed in the 1960s and widely used in early systems
- Focus on vowel classification as a fundamental building block for more complex speech recognition

### Potential Improvements

1. **Dynamic Feature Extraction**: Incorporate temporal dynamics by analyzing formant trajectories
2. **Advanced Classification**: Implement more sophisticated classifiers like GMMs or deep learning approaches
3. **Feature Engineering**: Explore additional acoustic features like spectral moments or MFCCs
4. **Data Augmentation**: Generate synthetic samples to improve classifier robustness

## License

This project is provided under the MIT License - see the LICENSE file for details.
