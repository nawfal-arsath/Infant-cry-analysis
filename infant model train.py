import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

# Function to extract spectrogram from audio
def audio_to_spectrogram(file_path, target_shape=(128, 128)):
    y, sr = librosa.load(file_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Ensure fixed size
    if mel_spec_db.shape[1] < target_shape[1]:  
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_shape[1] - mel_spec_db.shape[1])), mode='constant')
    elif mel_spec_db.shape[1] > target_shape[1]:  
        mel_spec_db = mel_spec_db[:, :target_shape[1]]

    return mel_spec_db

# Load dataset
dataset_path =r"E:\aug" 
X, y = [], []
classes = os.listdir(dataset_path)

for class_label in classes:
    class_folder = os.path.join(dataset_path, class_label)
    for file in os.listdir(class_folder):
        file_path = os.path.join(class_folder, file)
        spectrogram = audio_to_spectrogram(file_path)
        X.append(spectrogram)
        y.append(class_label)

X = np.array(X).reshape(-1, 128, 128, 1)  # Add channel dimension
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open("label_encoder2.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(classes))

# Augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Build the improved model
def build_model(input_shape=(128, 128, 1), num_classes=len(classes)):
    inputs = tf.keras.Input(shape=input_shape)

    # Apply Augmentation
    x = data_augmentation(inputs)

    # EfficientNet as feature extractor
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze base model
    x = tf.keras.layers.Conv2D(3, (3, 3), padding="same", activation="relu")(x)  # Convert grayscale to 3 channels
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # LSTM layer for sequential analysis
    x = tf.keras.layers.Reshape((1, -1))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)

    # Attention mechanism
    attention = tf.keras.layers.Dense(64, activation="tanh")(x)
    attention = tf.keras.layers.Dense(1, activation="softmax")(attention)
    x = tf.keras.layers.Multiply()([x, attention])

    # Fully connected layers
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Compile model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# Train the model
model = build_model()
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=20, batch_size=32, callbacks=[early_stopping])

# Save model
model.save("infant_cry_model2.keras")
