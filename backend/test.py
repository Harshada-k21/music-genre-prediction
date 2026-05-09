import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
import keras

# Waking up the AI
print("Waking up the AI...")
model = keras.models.load_model("music_genre_cnn_model.keras")
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

TEST_TRACK = r"C:\Users\Admin\Downloads\dua lipa - homesick (lyrics) [34Evv6yLbXQ].mp3" # <-- CHANGE THIS PATH

def test_full_track(file_path):
    print(f"Listening to the entire track: {file_path}...")
    
    # Load the ENTIRE audio file (duration=None)
    signal, sr = librosa.load(file_path, sr=22050, duration=None)
    
    # Extract MFCCs for the whole song
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=2048, n_mfcc=13, hop_length=512)
    mfcc = mfcc.T
    
    # The CNN expects chunks of 130 time steps (~3 seconds)
    expected_length = 130
    
    # Calculate how many full 3-second chunks we can get out of the song
    num_chunks = len(mfcc) // expected_length
    
    if num_chunks == 0:
        print("Audio clip too short! Needs to be at least 3 seconds.")
        return
        
    # Chop the song into chunks and store them in a list
    chunks = []
    for i in range(num_chunks):
        start = i * expected_length
        end = start + expected_length
        chunk = mfcc[start:end]
        chunks.append(chunk)
        
    # Convert the list to a 3D numpy array, then add the channel dimension for the CNN
    batch_features = np.array(chunks)
    batch_features = batch_features[..., np.newaxis] # Shape will be (num_chunks, 130, 13, 1)
    
    # Feed all chunks to the model at once!
    # This returns a probability distribution for EVERY chunk.
    predictions = model.predict(batch_features, verbose=0)
    
    # AVERAGE the predictions across all chunks to find the true overall genre
    average_prediction = np.mean(predictions, axis=0)
    
    # AVERAGE the predictions across all chunks to find the true overall genre
    average_prediction = np.mean(predictions, axis=0)
    
    # Get the indices of the top 3 predictions by sorting the array
    top_3_indices = np.argsort(average_prediction)[::-1][:3]
    
    print("\n" + "="*40)
    print(f"🎵 TRACK ANALYZED ({num_chunks} chunks)")
    print("🏆 TOP 3 PREDICTIONS:")
    
    for i in range(3):
        genre_index = top_3_indices[i]
        genre_name = GENRES[genre_index]
        confidence = average_prediction[genre_index] * 100
        
        # Add a little visual flair for the winner
        if i == 0:
            print(f"   #1: {genre_name.upper()} ({confidence:.2f}%)  <-- Winning Guess")
        else:
            print(f"   #{i+1}: {genre_name.capitalize()} ({confidence:.2f}%)")
            
    print("="*40 + "\n")