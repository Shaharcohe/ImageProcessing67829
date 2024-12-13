import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import numpy as np
from typing import Optional, Literal
from PIL import Image
import scipy.signal
from math import log
from scipy.io import wavfile
import scipy.signal


# -----------------------------
# Global Constants and Settings
# -----------------------------

SAMPLING_RATE = 44.1 * 1000  # Sampling rate in Hz (44.1 kHz)

PEAK_FREQ = 20000  # Target frequency for peak detection (20 kHz)
MAX_FREQ = 20000   # Maximum frequency for spectrogram display (20 kHz)

# Plot design settings
COLOR_THEME = 'turbo'  # Colormap used for spectrograms
FIG_SIZE = (8, 5)      # Default figure size for plots

# STFT properties
WINDOW_FUNCTION = 'hann'  # Window function used in STFT
WIN_LENGTH = 1102         # Window length for STFT
HOP_LENGTH = 441          # Hop length for STFT

# Watermark parameters
K = 2  # Number of intervals for watermark embedding
L = 2 ** round(log(SAMPLING_RATE * 5, 2))  # Length of each interval for watermark embedding

# File paths for tasks
task_1_path = 'Task 1/task1.wav'
task_2_paths = [f'Task 2/{i}_watermarked.wav' for i in range(9)]
task_3_paths = [f'Task 3/task3_watermarked_method{i}.wav' for i in [1, 2]]

# Watermark frequencies and amplitudes
FREQUENCIES = {
    'bad': 10000,       # 10 kHz frequency for bad watermark
    'good': 19.5 * 1000  # 19.5 kHz frequency for good watermark (beyond human hearing range)
}

AMPLITUDES = {
    'bad': 1e-1,  # Amplitude for bad watermark
    'good': 1e-3  # Amplitude for good watermark
}

# -----------------------------
# Function Definitions
# -----------------------------



def load_file(path, sr=None) -> np.ndarray:
    """
    This function loads a wav file and returns a numpy array containing the audio data.
    :param path: Path to audio file
    :param sr: Sampling Rate
    :return: audio data
    """
    if sr:
        y, sr = librosa.load(path, sr=sr)
    else:
        y, sr = librosa.load(path)
    return y, sr


def create_spectrogram(audio_data: np.ndarray,
                       sr: Optional[int] = None,
                       show: bool = True,
                       title_addition: Optional[str] = None,
                       output_path: Optional[str] = None) -> None:
    """
    This function creates a spectrogram for a given audio.
    :param audio_data:
    :param sr:
    :return:
    """
    plt.figure(figsize=FIG_SIZE)
    X = librosa.stft(audio_data, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, window=WINDOW_FUNCTION)
    librosa.display.specshow(librosa.amplitude_to_db(abs(X)), sr=sr, x_axis='time', y_axis='hz', cmap=COLOR_THEME)
    plt.colorbar(format="%+2.0f dB")
    title = "Spectrogram in Decibel Scale"
    if title_addition is not None:
        title += f" - {title_addition}"
    plt.title(title.title())
    if show:
        plt.show()
    if output_path is not None:
        plt.savefig(output_path)


def add_watermark(type: Literal['good', 'bad'], audio_data: np.ndarray, sr=SAMPLING_RATE):
    """
    Adds a watermark to the audio data based on the specified type.

    Parameters:
        type (str): Type of watermark ('good' or 'bad').
        audio_data (np.ndarray): Original audio data.
        sr (float): Sampling rate of the audio data.

    Returns:
        np.ndarray: Audio data with the watermark added.
    """
    duration = len(audio_data) / sr
    t = np.linspace(0, duration, len(audio_data))  # represents time
    if type == 'bad':
        watermark = AMPLITUDES[type] * np.sin(2 * np.pi * FREQUENCIES[type] * t)
        return audio_data + watermark
    else:
        new_audio = np.copy(audio_data)
        intervals = choose_random_intervals(len(audio_data), K, L)
        d = L
        freq = FREQUENCIES['good']
        while d > 0 and freq < (SAMPLING_RATE / 2 - 1):
            new_audio += add_interval_mark(
                audio_data,
                intervals=[int(start + L - d) for start in intervals],
                l=d,
                sr = SAMPLING_RATE,
                freq=freq
            )
            freq += 20
            d -= int(L / 100)
        return new_audio

def save_audio(output_path: str, audio_data: np.ndarray, sr) -> None:
    sf.write(output_path, audio_data, samplerate=sr)


def most_informative_intervals(orig_audio: np.ndarray, k, l):
    rolling_sum = np.convolve(orig_audio, np.ones(l, dtype=int), mode='valid')
    selected_indices = []
    for _ in range(k):
        max_index = np.argmax(rolling_sum)
        selected_indices.append(max_index)
        start = max_index
        end = max_index + l
        rolling_sum[start:end] = -np.inf

    return selected_indices

def add_interval_mark(data, intervals, l, sr, freq):
    watermark = np.zeros_like(data)
    duration = len(watermark) / sr
    t = np.linspace(0, duration, len(data))  # represents time
    for start in intervals:
        end = int(start + l)
        start = int(start)
        time = t[start:end]
        watermark[start:end] +=  AMPLITUDES['good'] * np.sin(2 * np.pi * freq * time)
    return watermark

def choose_random_intervals(N, k, l):
    """
    Selects k random intervals of length l within an audio signal of length N.

    Parameters:
        N (int): Length of the audio signal.
        k (int): Number of intervals to select.
        l (int): Length of each interval.

    Returns:
        list: List of starting indices for the selected intervals.
    """
    intervals = np.random.randint(low=0, high=N / l, size=k)
    intervals *= l
    return intervals


def load_image(filepath):
    """
    Loads an image as a grayscale NumPy array.

    Parameters:
        filepath (str): Path to the image file.

    Returns:
        image_array (2D np.ndarray): Grayscale image as a NumPy array.
    """
    image = Image.open(filepath).convert('L')  # Convert to grayscale ('L' mode)
    image_array = np.array(image)
    return image_array

def classify_audio_file(audio, sr):
    """
    Classifies an audio file based on the number of detected peaks at a target frequency.

    Parameters:
        audio (np.ndarray): Audio data to classify.
        sr (int): Sampling rate of the audio data.

    Returns:
        int or None: Classification label (1, 2, 3) or None if no class is detected.
    """
    target_frequency = PEAK_FREQ
    f, t, Zxx = scipy.signal.stft(audio, fs=sr)
    freq_idx = np.argmin(np.abs(f - target_frequency))

    Zxx_20 = np.abs(Zxx[freq_idx, :])
    sigma = 7
    Zxx_20_smoothed = scipy.ndimage.gaussian_filter1d(Zxx_20, sigma=sigma)
    peaks, _ = scipy.signal.find_peaks(Zxx_20_smoothed, width=70, prominence = 1e-5)
    num_peaks = len(peaks)
    if num_peaks == 11:
        return 1
    if num_peaks == 15:
        return 2
    if num_peaks == 19:
        return 3
    else:
        return None

def compute_freq_median(audio, sr):
    """
    Computes the median frequency of an audio signal using the Short-Time Fourier Transform (STFT).

    Parameters:
        audio (np.ndarray): The audio data array.
        sr (int): The sampling rate of the audio signal in Hz.

    Returns:
        float: The median frequency of the audio signal.

    Steps:
        1. Compute the STFT of the audio signal with a window size of 1024 samples.
        2. Calculate the magnitude (absolute values) of the STFT coefficients.
        3. For each time frame, find the frequency corresponding to the maximum magnitude.
        4. Return the median of these frequencies.
    """
    f, t, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=1024)
    S = np.abs(Zxx)
    freq = f[np.argmax(S, axis=0)]
    return np.median(freq)

def determine_speedup_method(print_results=True):
    """
    Determines the speedup or slowdown method used for two audio files by comparing their median frequencies.

    Parameters:
        print_results (bool): If True, prints the median frequencies and the speedup/slowdown ratio.

    Steps:
        1. Load two audio files specified in the global `task_3_paths` list.
        2. Compute and display spectrograms for both audio files.
        3. Calculate the median frequency for each audio file using `compute_freq_median`.
        4. Compute the speedup/slowdown ratio by dividing the median frequency of the second file by the first.
        5. Optionally, print the results and save the spectrogram plot to a file.
    """
    audio_arrays = []
    sample_rates = []
    for path in task_3_paths:
        sr, data = wavfile.read(path)
        audio_arrays.append(data)
        sample_rates.append(sr)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(FIG_SIZE[0] * 2.5, FIG_SIZE[1]))

    for i in range(len(task_3_paths)):
        Pxx, freqs, bins, im = ax[i].specgram(audio_arrays[i], Fs=sample_rates[i], NFFT=1024, noverlap=512,
                                              cmap='magma')
        ax[i].set_xlabel('Time (s)')
        ax[i].set_ylabel('Frequency (Hz)')
        ax[i].set_title(f'Spectrogram Method {i + 1}')
        ax[i].set_ylim(0, MAX_FREQ)  # Scale the y-axis to (0, 20000) Hz

    median_freq1 = compute_freq_median(audio_arrays[0], sample_rates[0])
    median_freq2 = compute_freq_median(audio_arrays[1], sample_rates[1])
    speedup_ratio = median_freq2 / median_freq1

    if print_results:
        print(f"Median Frequency (Method 1): {median_freq1} Hz")
        print(f"Median Frequency (Method 2): {median_freq2} Hz")
        print(f"Speedup/Slowdown Ratio (x): {speedup_ratio}")

    plt.savefig(f"plots/task3.png")


def main():
    # 1. Add bad and good watermarks to the Task 1 file and save them
    audio_data, sr = load_file(task_1_path, sr=SAMPLING_RATE)

    # Add bad watermark and save
    bad_watermarked = add_watermark('bad', audio_data, sr)
    save_audio('Task 1/task1_bad_watermarked.wav', bad_watermarked, sr)
    print("Saved Task 1 with bad watermark as 'Task 1/task1_bad_watermarked.wav'.")

    # Add good watermark and save
    good_watermarked = add_watermark('good', audio_data, sr)
    save_audio('Task 1/task1_good_watermarked.wav', good_watermarked, sr)
    print("Saved Task 1 with good watermark as 'Task 1/task1_good_watermarked.wav'.")

    # 2. Classify Class 2 files and print the classification results
    print("\nClassifying Task 2 files:")
    for path in task_2_paths:
        audio, sr = load_file(path, sr=SAMPLING_RATE)
        classification = classify_audio_file(audio, sr)
        print(f"{path}: Classified as Class {classification}")

    # 3. Perform Task 3: Determine speedup method and display results
    print("\nPerforming Task 3:")
    determine_speedup_method(print_results=True)
    print("Task 3 spectrograms and results saved.")

if __name__ == "__main__":
    main()