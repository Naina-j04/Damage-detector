import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model


DATASET_DIR = "simple_dataset"  
MODEL_DIR = "models"
MODEL_NAME = "simple_damage_classifier.keras"
CLASS_INDEX_FILE = "class_indices.json"
IMG_SIZE = (224, 224)  # Increased image size for better feature extraction
BATCH_SIZE = 32
EPOCHS = 50  # Increased epochs with early stopping
LEARNING_RATE = 1e-4

# Prepare dataset
os.makedirs(MODEL_DIR, exist_ok=True)

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Load pre-trained MobileNetV2 as base model
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze the base model layers
base_model.trainable = False

# Build model on top of MobileNetV2
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, MODEL_NAME),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Fine-tuning phase
print("\nStarting fine-tuning phase...")
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile model with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Continue training
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)

# Save class indices
class_indices_path = os.path.join(MODEL_DIR, CLASS_INDEX_FILE)
with open(class_indices_path, "w") as f:
    json.dump(train_generator.class_indices, f)

print("✅ Model training complete!")

# Print final metrics
final_metrics = model.evaluate(val_generator)
print("\nFinal Validation Metrics:")
print(f"Loss: {final_metrics[0]:.4f}")
print(f"Accuracy: {final_metrics[1]:.4f}")
print(f"AUC: {final_metrics[2]:.4f}")
print(f"Precision: {final_metrics[3]:.4f}")
print(f"Recall: {final_metrics[4]:.4f}")
