import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the saved model
model = tf.keras.models.load_model('dog_cat_model.h5')

def predict_image(img):
    # Resize and normalize the image to match the training data
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_reshaped = np.expand_dims(img_normalized, axis=0)
    
    # Generate prediction
    prediction = model.predict(img_reshaped)[0][0]
    
    # Class 0 is Cat, Class 1 is Dog based on alphabetical folder names
    prob_dog = float(prediction)
    prob_cat = float(1 - prediction)
    
    return {"Dog": prob_dog, "Cat": prob_cat}

# Build the web interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="CNN Dog vs Cat Classifier",
    description="Upload an image of a pet to test the model."
)

if __name__ == "__main__":
    interface.launch()