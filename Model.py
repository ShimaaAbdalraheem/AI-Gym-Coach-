import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from DataPreprocessing import DataPreprocessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, f1_score

# usage
data_dir = "exercise_data"
preprocessor = DataPreprocessor(data_dir)
X, y, exercise_labels = preprocessor.preprocess_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X to have the correct shape for Conv2D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# Define the number of classes
#NUM_CLASSES = len(exercise_labels)
NUM_CLASSES = 6

# Build a more complex CNN model using functional API
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2], 1))
x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = tf.keras.layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Compile the model with a learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[lr_schedule, early_stopping])

model.summary()


# Evaluate the model on training and test data
exercise_Labels = {'Side Plank true': 0, 'Upper Crunches true': 1, 'Wrong stationary lunges': 2, 'Wrong Side Plank': 3, 'Wrong Upper Crunches': 4, 'stationary lunges true': 5}

# Make predictions on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute accuracy for training and testing sets
train_accuracy = accuracy_score(y_train, np.argmax(y_train_pred, axis=1))
test_accuracy = accuracy_score(y_test, np.argmax(y_test_pred, axis=1))
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Compute precision, recall, and F1-score for testing data
precision_test = precision_score(y_test, np.argmax(y_test_pred, axis=1), average='weighted')
recall_test = recall_score(y_test, np.argmax(y_test_pred, axis=1), average='weighted')
f1_test = f1_score(y_test, np.argmax(y_test_pred, axis=1), average='weighted')
print("Precision (Testing):", precision_test)
print("Recall (Testing):", recall_test)
print("F1-score (Testing):", f1_test)

# Compute precision, recall, and F1-score for training data
precision_train = precision_score(y_train, np.argmax(y_train_pred, axis=1), average='weighted')
recall_train = recall_score(y_train, np.argmax(y_train_pred, axis=1), average='weighted')
f1_train = f1_score(y_train, np.argmax(y_train_pred, axis=1), average='weighted')
print("Precision (Training):", precision_train)
print("Recall (Training):", recall_train)
print("F1-score (Training):", f1_train)


# Compute confusion matrix for testing data
conf_matrix_test = confusion_matrix(y_test, np.argmax(y_test_pred, axis=1))
print("Confusion Matrix (Testing):")
print(conf_matrix_test)

# Compute confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train, np.argmax(y_train_pred, axis=1))
print("Confusion Matrix (Training):")
print(conf_matrix_train)

# Plot confusion matrix for training data
plt.figure(figsize=(18, 16))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=exercise_Labels.keys(), yticklabels=exercise_Labels.keys())
plt.title('Confusion Matrix - Training Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_training.png')  
plt.show()

# Plot confusion matrix for testing data
plt.figure(figsize=(18, 16))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=exercise_Labels.keys(), yticklabels=exercise_Labels.keys())
plt.title('Confusion Matrix - Testing Data')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_testing.png')  
plt.show()

# Plot training and testing loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and testing accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
# Save the plot
plt.savefig('training_testing_metrics.png')
plt.show()

# Save the trained model
model.save("exercise_classifier_model_v2.keras")
print("Model saved as exercise_classifier_model_v2.keras")
