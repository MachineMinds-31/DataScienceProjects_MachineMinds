import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import Audio
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder, label_binarize
from joblib import dump, load

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#loading actual dataset
paths = []
labels = []
for dirname, _, filenames in os.walk('/content/drive/MyDrive/CLG_PROJECT/TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]  #to store the label such as fear happy sad
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')

df = pd.DataFrame()
df['speech'] = paths #input as speech
df['label'] = labels #output as label
df.head()

df = pd.DataFrame({'speech': paths, 'label': labels})

# Print some information to diagnose the issue
print("Number of entries in df:", len(df))
print("Unique labels in df:", df['label'].unique())

# Visualize class distribution
sns.countplot(data=df, x='label')  # 'x' specifies the column to count

# Optionally, you can rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Show the plot
plt.show()

# Extract features
def extract_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

X_mfcc = df['speech'].apply(extract_features)
X = np.array(X_mfcc.tolist())

# Apply data augmentation
# Implement your data augmentation techniques here if needed

# Expand dimensions for Conv1D input
X = np.expand_dims(X, -1)

# Encode labels
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']]).toarray()

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(40, 1)),
    MaxPooling1D(2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    BatchNormalization(),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[reduce_lr, checkpoint])

# Evaluate on validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Flatten the enc.categories_ list
categories_flat = [label for sublist in enc.categories_ for label in sublist]

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories_flat, yticklabels=categories_flat)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Calculate and print precision, recall, F1 score, and accuracy for validation set
precision = metrics.precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = metrics.recall_score(y_true_classes, y_pred_classes, average='weighted')
f1_score = metrics.f1_score(y_true_classes, y_pred_classes, average='weighted')
accuracy = metrics.accuracy_score(y_true_classes, y_pred_classes)

print(f'Validation Precision: {precision:.4f}')
print(f'Validation Recall: {recall:.4f}')
print(f'Validation F1 Score: {f1_score:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

# ROC Curve
# Binarize the labels
y_bin = label_binarize(y_true_classes, classes=range(7))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(7):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(7):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"], label='Micro-average (AUC = {0:0.2f})', color='deeppink', linestyle=':', linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Save the model and encoder
model.save('optimized_model.h5')
dump(enc, 'encoder.joblib')

# Prediction Script
def predict_emotion(audio_file_path):
    # Load the model
    loaded_model = tf.keras.models.load_model('optimized_model.h5')
    # Load the encoder
    enc = load('encoder.joblib')

    # Extract features
    X_test = np.expand_dims(extract_features(audio_file_path), -1)
    X_test = np.expand_dims(X_test, 0)

    # Make predictions
    pred = loaded_model.predict(X_test)
    y_pred = enc.inverse_transform(pred)

    print(f'Predicted Emotion: {y_pred.flatten()[0]}')

# Example usage of the prediction script
audio_file_path = '/content/drive/MyDrive/CLG_PROJECT/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/OAF_Fear/OAF_back_fear.wav'
predict_emotion(audio_file_path)