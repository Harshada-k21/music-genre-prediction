import os
import librosa
import math
import json

# Paths
DATASET_PATH = "genres_original"
JSON_PATH = "data.json"

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """
    Extracts MFCCs from the audio dataset and saves them into a JSON file.
    We split the 30-second tracks into smaller segments to give our AI more data to learn from!
    """
    
    # Dictionary to store our data
    data = {
        "mapping": [], # Maps the genre string (e.g., "pop") to an integer (e.g., 0)
        "labels": [],  # The target output for our AI
        "mfcc": []     # The input data for our AI
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Loop through all genre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure we are not at the root level
        if dirpath is not dataset_path:
            
            # Save the genre label (i.e., the sub-folder name)
            dirpath_components = dirpath.split("/") # use "\\" if on Windows
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing Genre: {semantic_label}")

            # Process files for a specific genre
            for f in filenames:
                
                # Load the audio file
                file_path = os.path.join(dirpath, f)
                try:
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                except Exception as e:
                    print(f"Skipping corrupted file: {file_path}")
                    continue

                # Process segments by extracting MFCCs
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], 
                                                sr=sr, 
                                                n_fft=n_fft, 
                                                n_mfcc=n_mfcc, 
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # Store MFCC for segment if it has the expected length
                    if len(mfcc) == expected_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)

    # Save the huge dictionary to a JSON file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    print("Data successfully saved to data.json!")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)