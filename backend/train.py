import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras

DATA_PATH = "data.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # *** NEW: CNNs require a 3D array (width, height, channels) ***
    # Our data is currently 2D (time, mfcc). We need to add a "channel" dimension.
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Build the CNN Architecture
    model = keras.Sequential([
        # 1st Conv Layer (Scans the audio for basic patterns)
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(), # Helps the network train faster and more stably

        # 2nd Conv Layer (Looks for more complex patterns)
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        # 3rd Conv Layer
        keras.layers.Conv2D(32, (2, 2), activation='relu'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        keras.layers.BatchNormalization(),

        # Flatten the final 2D results and feed them into a standard layer
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),

        # Output Layer (10 genres)
        keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    print("Starting CNN training...")
    # I set epochs to 150. CNNs learn much faster and more efficiently than MLPs!
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=150)

    model.save("music_genre_cnn_model.keras")
    print("CNN Model saved as music_genre_cnn_model.keras!")