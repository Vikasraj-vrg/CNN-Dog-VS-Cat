import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Define folder paths based on your new dataset
train_dir = 'training_set'
test_dir = 'test_set'

batch_size = 32
img_height = 128
img_width = 128

# Load Training Data
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Load Testing/Validation Data
val_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# 2. Define the CNN Architecture
model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
  layers.MaxPooling2D(2, 2),
  layers.Conv2D(64, (3, 3), activation='relu'),
  layers.MaxPooling2D(2, 2),
  layers.Conv2D(128, (3, 3), activation='relu'),
  layers.MaxPooling2D(2, 2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5), # Prevents overfitting
  layers.Dense(1, activation='sigmoid') # Binary classification
])

# 3. Compile and Train
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Starting training...")
epochs = 15 # Increased slightly since this dataset is well-structured
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# 4. Save the Model
model.save('dog_cat_model.h5')
print("Model saved as dog_cat_model.h5")