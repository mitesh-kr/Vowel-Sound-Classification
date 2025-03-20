# -*- coding: utf-8 -*-
"""
Feature extraction module for vowel classification system
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import find_peaks

from preprocess import audio_preprocess

def extract_formants_lpc(frame, sr, order=14):
    """
    Extract formants using Linear Predictive Coding (LPC).
    
    Parameters:
    -----------
    frame : ndarray
        Audio frame
    sr : int
        Sample rate
    order : int, optional
        LPC order, default is 14
    
    Returns:
    --------
    formants : ndarray
        Array of formant frequencies (F1, F2, F3)
    """
    # Skip silent frames
    if np.mean(np.abs(frame)) < 1e-3:
        return np.array([np.nan, np.nan, np.nan])

    # Add small noise to avoid numerical issues
    a = librosa.lpc(frame + 1e-6 * np.random.randn(*frame.shape), order=order)
    
    # Normalize LPC coefficients
    a = a / np.max(np.abs(a))
    
    # Find roots of the LPC polynomial
    roots = np.roots(a)
    
    # Keep only roots with positive imaginary part (frequency domain)
    roots = roots[np.imag(roots) > 0]
    
    # Remove roots close to unit circle (potential instability)
    roots = roots[np.abs(roots) < 0.99]
    
    # Convert to angles and then to frequencies
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    
    # Sort frequencies to get formants
    formants = np.sort(freqs)

    # Return first three formants or pad with NaN if fewer than 3
    return formants[:3] if len(formants) >= 3 else np.pad(formants, (0, 3 - len(formants)), 'constant', constant_values=np.nan)

def extract_f0_autocorr(frame, sr, min_freq=50, max_freq=500):
    """
    Extract fundamental frequency using improved autocorrelation with peak detection.
    
    Parameters:
    -----------
    frame : ndarray
        Audio frame
    sr : int
        Sample rate
    min_freq : int, optional
        Minimum frequency to consider, default is 50 Hz
    max_freq : int, optional
        Maximum frequency to consider, default is 500 Hz
    
    Returns:
    --------
    f0 : float
        Fundamental frequency or NaN if not found
    """
    # Skip silent frames
    if np.mean(np.abs(frame)) < 1e-3:
        return np.nan

    # Compute autocorrelation
    corr = np.correlate(frame, frame, mode='full')[len(frame)//2:]
    
    # Convert frequency limits to lag indices
    min_lag = int(sr / max_freq)
    max_lag = min(int(sr / min_freq), len(corr) - 1)

    # Check if valid lags exist
    if min_lag >= len(corr) or max_lag >= len(corr):
        return np.nan

    # Find peaks in autocorrelation
    peaks, _ = find_peaks(corr[min_lag:max_lag])

    # Return NaN if no peaks found
    if len(peaks) == 0:
        return np.nan

    # Calculate F0 from first peak
    peak_idx = min_lag + peaks[0]
    f0 = sr / peak_idx if peak_idx > 0 else np.nan
    
    return f0

def extract_formant_frequencies(audio_file, sr=44100, frame_size=25, frame_stride=10, order=12):
    """
    Extract fundamental frequency (F0) and formant frequencies (F1, F2, F3) from an audio file.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file
    sr : int, optional
        Sample rate, default is 44100 Hz
    frame_size : int, optional
        Frame size in milliseconds, default is 25 ms
    frame_stride : int, optional
        Frame stride in milliseconds, default is 10 ms
    order : int, optional
        LPC order, default is 12
    
    Returns:
    --------
    features : dict
        Dictionary containing F0, F1, F2, and F3 values
    """
    # Preprocess audio
    windowed_frames, sr = audio_preprocess(audio_file, sr, frame_size, frame_stride)

    formants_list = []
    f0_list = []

    # Process each frame
    for frame in windowed_frames:
        # Extract formants
        formants = extract_formants_lpc(frame, sr, order)
        if not np.isnan(formants).any():
            formants_list.append(formants)

        # Extract F0
        f0 = extract_f0_autocorr(frame, sr)
        if not np.isnan(f0):
            f0_list.append(f0)

    # Calculate mean values
    f1, f2, f3 = np.nanmean(formants_list, axis=0) if formants_list else (np.nan, np.nan, np.nan)
    f0 = np.nanmean(f0_list) if f0_list else np.nan

    return {'F0': f0, 'F1': f1, 'F2': f2, 'F3': f3}

def load_and_extract_features(dataset_path):
    """
    Load the vowel dataset and extract formant frequencies.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the dataset directory
    
    Returns:
    --------
    features_df : pandas.DataFrame
        DataFrame containing extracted features for all samples
    """
    categories = ['Male', 'Female']
    vowels = ['a', 'e', 'i', 'o', 'u']
    results = []

    for category in categories:
        category_path = os.path.join(dataset_path, category)

        if not os.path.isdir(category_path):
            print(f"Warning: Path not found: {category_path}")
            continue

        for vowel in vowels:
            vowel_path = os.path.join(category_path, vowel)

            if not os.path.isdir(vowel_path):
                print(f"Warning: Path not found: {vowel_path}")
                continue

            files = [f for f in os.listdir(vowel_path) if f.endswith('.wav')]

            for file in files:
                file_path = os.path.join(vowel_path, file)

                try:
                    features = extract_formant_frequencies(file_path)
                    results.append({
                        'file_path': file_path,
                        'category': category,
                        'vowel': vowel,
                        'filename': file,
                        'F0': features['F0'],
                        'F1': features['F1'],
                        'F2': features['F2'],
                        'F3': features['F3']
                    })

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    return pd.DataFrame(results)
