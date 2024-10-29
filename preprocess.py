import math
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import librosa
import kagglehub
import keras
import os
import csv
from dataclasses import dataclass
from dataclass_csv import DataclassReader
from tqdm import tqdm

MUSICNET_PATH = "musicnet/musicnet"
OUTPUT_PATH = "processed_musicnet"
# Set the resolution to 1/8th of a quarter note beat, which is equivalent to a thirty-second note
BEAT_RESOLUTION = 1 / 8
# Set the number of different notes encoded in the MIDI format
MIDI_NOTES = 128


@dataclass
class MusicnetTuple:
    start_time: int
    end_time: int
    instrument: int
    note: int
    start_beat: float
    end_beat: float
    note_value: str


def get_start_beat_index(note_tuple):
    # Get ceiling of the start beat divided by the resolution to get its related index in our map
    return math.ceil(note_tuple.start_beat / BEAT_RESOLUTION)


def get_end_beat_index(note_tuple):
    # Acquire the note's actual end beat by adding the start_beat and end_beat (offset) values
    # Get floor of the end beat divided by the resolution to get its related index in our map
    return math.floor((note_tuple.start_beat + note_tuple.end_beat) / BEAT_RESOLUTION)


def create_note_map(data):
    # Find the note that ends the latest. This allows us to determine the endpoint of the note map
    final_note = max(data, key=lambda note: note.start_beat + note.end_beat)

    # Note that the end_beat field denotes the *duration* of the note, not the actual point where the note ends
    # Calculate the final index of our note_map
    endpoint = get_end_beat_index(final_note)

    # Define the note map as a two-dimensional list of boolean values denoting whether a note is played at a particular
    # beat point. The minimum note value is a thirty-second, and there are 128 potential notes that can be played at any
    # given time. The 129th note represents the point where the soundtrack ends
    note_map = [[0] * (MIDI_NOTES + 1) for _ in range(endpoint + 1)]

    # Sort the note list by start point
    data.sort(key=lambda note: note.start_beat)

    # Iterate over every note and mark it down as played according to duration
    for note_tuple in data:
        start_beat_index = get_start_beat_index(note_tuple)
        end_beat_index = get_end_beat_index(note_tuple)
        # Iterate over every point where the note is played and set the value at the corresponding index to 1
        for beat_point in range(start_beat_index, end_beat_index + 1):
            note_map[beat_point][note_tuple.note] = 1

    # Denote that the track has ended
    note_map[endpoint][MIDI_NOTES] = 1
    return note_map


def get_processed_array(file_path):
    with open(file_path, newline='') as file:
        # Insert data into a list of tuples
        data = list(DataclassReader(file, MusicnetTuple))
        # Generate note map
        note_map = create_note_map(data)
        # Convert map into a two-dimensional NumPy array
        return np.array(note_map)


if __name__ == "__main__":
    try:
        # Create target directories
        os.makedirs(f"{OUTPUT_PATH}/train_labels")
        os.makedirs(f"{OUTPUT_PATH}/test_labels")
        os.makedirs(f"{OUTPUT_PATH}/train_data")
        os.makedirs(f"{OUTPUT_PATH}/test_data")
    except FileExistsError:
        print("Target directory already exists!")
        quit()

    # Collect paths of train and test set labels

    train_label_paths = glob.glob(f"{MUSICNET_PATH}/train_labels/*.csv")
    test_label_paths = glob.glob(f"{MUSICNET_PATH}/test_labels/*.csv")

    for train_label_path in tqdm(train_label_paths):
        # Process file
        output_array = get_processed_array(train_label_path)
        # Output array object with the same name as the .csv
        name_stem = pathlib.PurePath(train_label_path).stem
        np.save(f"{OUTPUT_PATH}/train_labels/{name_stem}.npy", output_array)

    # Repeat process for test data
    for test_label_path in tqdm(test_label_paths):
        output_array = get_processed_array(test_label_path)

        name_stem = pathlib.PurePath(test_label_path).stem
        np.save(f"{OUTPUT_PATH}/test_labels/{name_stem}.npy", output_array)

    # Collect paths of train and test set tracks

    train_data_paths = glob.glob(f"{MUSICNET_PATH}/train_data/*.wav")
    test_data_paths = glob.glob(f"{MUSICNET_PATH}/test_data/*.wav")

    # Convert tracks into spectrograms and save as .npy objects

    for train_data_path in tqdm(train_data_paths):
        audio, sr = librosa.load(train_data_path, sr=None)
        # Convert audio to magnitude spectrogram
        spectrogram = abs(librosa.stft(audio))

        name_stem = pathlib.PurePath(train_data_path).stem
        np.save(f"{OUTPUT_PATH}/train_data/{name_stem}.npy", spectrogram)

    for test_data_path in tqdm(test_data_paths):
        audio, sr = librosa.load(test_data_path, sr=None)
        # Convert audio to magnitude spectrogram
        spectrogram = abs(librosa.stft(audio))

        name_stem = pathlib.PurePath(test_data_path).stem
        np.save(f"{OUTPUT_PATH}/test_data/{name_stem}.npy", spectrogram)
