# -*- coding: utf-8 -*-
"""
Preprocessing module for vowel classification system
"""

import numpy as np
import librosa

def audio_preprocess(file_path, sr=44100, frame_size=25, frame_stride=10):
    """
    Load, preprocess, and frame an audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    sr : int, optional
        Sample rate, default is 44100 Hz
    frame_size : int, optional
        Frame size in milliseconds, default is 25 ms
    frame_stride : int, optional
        Frame stride in milliseconds, default is 10 ms
    
    Returns:
    --------
    windowed_frames : ndarray
        Array of framed and windowed audio data
    sr : int
        Sample rate
    """
    # Load audio file
    audio, sr = librosa.load(file_path, sr=sr)

    # Apply pre-emphasis filter
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Calculate frame parameters
    frame_length = int(sr * frame_size / 1000)
    frame_step = int(sr * frame_stride / 1000)
    signal_length = len(audio)
    num_frames = max(1, 1 + int(np.ceil((signal_length - frame_length) / frame_step)))

    # Pad signal to ensure all frames have equal length
    pad_length = (num_frames - 1) * frame_step + frame_length
    pad_signal = np.append(audio, np.zeros(pad_length - signal_length))

    # Create indices for framing
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    # Extract frames
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Apply Hamming window
    windowed_frames = frames * np.hamming(frames.shape[1])

    return windowed_frames, sr
