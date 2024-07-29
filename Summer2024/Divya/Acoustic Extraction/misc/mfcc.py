import librosa
import pandas as pd
import numpy as np
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_filename, wav_filename):
    audio = AudioSegment.from_mp3(mp3_filename)
    audio.export(wav_filename, format="wav")


# Load the table
filename = "data_info_to_append_without_values"
table = pd.read_csv(f"{filename}.csv")

# Convert mp3 to wav
mp3_filename = "CA1HA_87.mp3"
wav_filename = "audio.wav"
convert_mp3_to_wav(mp3_filename, wav_filename)

# Load the .wav file using librosa
y, sr = librosa.load(wav_filename, sr=None)

# Initialize columns in the table for the extracted features
table['mfcc_1'] = np.nan
table['mfcc_2'] = np.nan
table['mfcc_3'] = np.nan

print("Start")

for i, row in table.iterrows():
    time = row['time']

    # Ensure the time value is within the audio duration
    if time > len(y) / sr:
        print(f"Time value {time} exceeds the duration of the audio.")
        continue

    # Extract MFCCs at the specified time
    frame_index = int(time * sr)
    if frame_index >= len(y):
        print(f"Frame index {frame_index} is out of bounds for time {time}.")
        continue

    # Extract a short segment of the audio around the specified time
    hop_length = 512  # hop length for STFT
    n_fft = 2048  # length of the FFT window
    mfccs = librosa.feature.mfcc(y=y[frame_index:frame_index + n_fft], sr=sr, n_mfcc=13, hop_length=hop_length)

    # Check if enough MFCC frames were extracted
    if mfccs.shape[1] == 0:
        print(f"No MFCC frames extracted for time {time}.")
        continue

    mfcc_1 = mfccs[0, 0]
    mfcc_2 = mfccs[1, 0]
    mfcc_3 = mfccs[2, 0]

    print(mfcc_1, mfcc_2, mfcc_3)

    # Insert values into the table
    table.at[i, 'mfcc_1'] = mfcc_1
    table.at[i, 'mfcc_2'] = mfcc_2
    table.at[i, 'mfcc_3'] = mfcc_3

    break

# Save the updated table if needed
# table.to_csv(f"{filename}_updated.csv", index=False)
