import os
import cv2
import numpy as np
from PIL import Image
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Apply bilateral filter to the image
def bilateral_filter(img):
    """Applies bilateral filter to reduce noise and keep edges sharp."""
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Augmentation function: you can expand this for more augmentations
def augment_frame(img):
    """Applies augmentations to the image."""
    # Example augmentation: Random flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)  # Flip horizontally
    return img

# Function to extract class name from the folder structure
def get_class_name_from_filename(filename):
    """Extracts class name from the folder name."""
    return os.path.basename(os.path.dirname(filename))

# Resize and augment frames before saving them in subfolders
def resize_and_augment_frames(input_path, output_path, size=224):
    """Resize and apply augmentation to images before saving them."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over all images in the input path (recursively)
    for filename in glob.glob(input_path + '/**/*.jpeg', recursive=True):
        # Open image
        img = Image.open(filename)
        img_np = np.array(img)  # Convert to numpy array

        # Apply bilateral filter
        img_filtered = bilateral_filter(img_np)

        # Resize image
        img_resized = cv2.resize(img_filtered, (size, size))  # Resize to 224x224
        
        # Apply augmentations
        augmented_img_np = augment_frame(img_resized)
        
        # Convert back to PIL Image after augmentation
        augmented_img = Image.fromarray(augmented_img_np)

        # Extract class name from folder name
        class_name = get_class_name_from_filename(filename)
        
        # Ensure class folder exists in output path
        class_folder = os.path.join(output_path, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        # Save the augmented image in the respective class subfolder
        name = os.path.split(filename)[1]
        save_path = os.path.join(class_folder, name)
        augmented_img.save(save_path)

        print(f"Saved: {save_path}")  # Debugging print statement

# Create the CNN model
def create_model():
    """Create a simple CNN model for training."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(8, activation='softmax')  # Assuming 8 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model using images from the directory
def train_model(train_dir, model):
    """Train the model using the images from the given directory."""
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=30, zoom_range=0.2)

    # Check the classes in the directory for debugging
    print("Classes in resized folder:", os.listdir(train_dir))

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')

    if not train_generator.class_indices:
        raise ValueError("No classes found in the directory")

    print(f"Found {train_generator.samples} images in {train_generator.num_classes} classes.")

    # Train the model
    model.fit(train_generator, epochs=10)

# Main function to start the process
if __name__ == "__main__":
    resized_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/ISL_Dataset 6.42.04 PM'  # Path to the resized frames folder
    output_folder = 'resized_folder'  # Path where the images will be saved

    # Resize and augment images
    resize_and_augment_frames(resized_folder, output_folder)

    # Create and train the model
    model = create_model()
    train_model(output_folder, model)
