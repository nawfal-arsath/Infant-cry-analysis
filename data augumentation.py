import librosa
import numpy as np
import soundfile as sf
import os
import random

# Directory paths
INPUT_DIR = r"C:\Users\nawfa\Downloads\infant cry dataset\donateacry_corpus"
OUTPUT_DIR = r"E:\aug"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_audio(file_path):
    """Load an audio file ensuring it's mono and has the correct data type."""
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    return audio.astype(np.float32), sr  # Ensure correct data type

def time_stretch(audio, rate=1.2):
    """Applies time stretching to an audio signal."""
    return librosa.effects.time_stretch(y=audio, rate=rate)  # Explicitly use 'y='

def pitch_shift(audio, sr, n_steps=2):
    """Applies pitch shifting."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)  # Explicitly use 'y='

def add_noise(audio, noise_level=0.005):
    """Adds random noise to the audio signal."""
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise

def get_class_counts():
    """Returns the number of samples in each class."""
    class_counts = {}
    for root, _, files in os.walk(INPUT_DIR):
        category = os.path.basename(root)
        num_files = len([file for file in files if file.endswith(".wav")])
        if num_files > 0:
            class_counts[category] = num_files
    return class_counts

def augment_audio(file_path, category, required_samples):
    """Applies multiple augmentations and saves results to balance dataset."""
    print(f"Processing: {file_path}")

    # Load audio
    audio, sr = load_audio(file_path)

    # Create category folder in OUTPUT_DIR if not exists
    category_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_path, exist_ok=True)

    # Save original file
    base_name = os.path.basename(file_path).replace(".wav", "")
    sf.write(os.path.join(category_path, f"{base_name}.wav"), audio, sr)

    # Apply augmentations multiple times to balance dataset
    augmentations = [time_stretch, pitch_shift, add_noise]
    generated_count = 0

    while generated_count < required_samples:
        aug_type = random.choice(augmentations)  # Random augmentation
        if aug_type == pitch_shift:
            aug_audio = aug_type(audio, sr, n_steps=random.choice([2, -2]))
        elif aug_type == time_stretch:
            aug_audio = aug_type(audio, rate=random.uniform(0.8, 1.2))
        else:
            aug_audio = aug_type(audio)

        sf.write(os.path.join(category_path, f"{base_name}_aug{generated_count}.wav"), aug_audio, sr)
        generated_count += 1

    print(f"Generated {generated_count} new files for {category}")

def balance_dataset():
    """Processes all audio files and balances the dataset across classes."""
    class_counts = get_class_counts()
    max_samples = max(class_counts.values())  # Find the largest class

    print(f"Class counts before augmentation: {class_counts}")
    print(f"Target samples per class: {max_samples}")

    for root, _, files in os.walk(INPUT_DIR):
        category = os.path.basename(root)
        num_files = class_counts.get(category, 0)

        if num_files == 0:
            continue

        required_samples = max_samples - num_files
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                augment_audio(file_path, category, required_samples)
                break  # Avoid excessive augmentation

    print("Dataset balancing complete!")

# Run the balancing process
if __name__ == "__main__":
    balance_dataset()
