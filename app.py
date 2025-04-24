# import gradio as gr
# import tensorflow as tf
# import numpy as np

# # Load your trained model
# model = tf.keras.models.load_model('plant_disease_model.h5')

# # List of classes
# class_names = [
#     'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
#     'Apple___healthy', 'Blueberry___healthy',
#     'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#     'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
#     'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
#     'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
#     'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
#     'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
#     'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
#     'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#     'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#     'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#     'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
# ]

# # Define the prediction function
# def predict(image):
#     image = image.resize((224, 224))  # Resize to model's expected size
#     image = np.array(image) / 255.0    # Normalize
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     predictions = model.predict(image)
#     predicted_class = class_names[np.argmax(predictions)]
#     confidence = np.max(predictions)
#     return f"{predicted_class} ({confidence*100:.2f}%)"

# # Build Gradio Interface
# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.Label(num_top_classes=5),
#     title="🌿 Agroshield-Plant Disease Detection",
#     description="Upload a plant leaf image and get the predicted disease class. Model trained on PlantVillage dataset."
# )

# # Launch app
# interface.launch()




import gradio as gr
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Class names
class_names = [...]  # Keep existing list

# Prediction function
def predict(image):
    # Existing prediction logic
    return f"{predicted_class} ({confidence*100:.2f}%)"

# Custom CSS matching web app
custom_css = """
:root {
    --primary: #2e7d32;
    --primary-hover: #1b5e20;
    --surface: #f5fbf6;
    --border: #c8e6c9;
}

body {
    background: var(--surface);
    font-family: 'Inter', system-ui;
}

.gradio-container {
    max-width: 800px;
    margin: 2rem auto;
    padding: 2.5rem;
    background: white;
    border-radius: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    border: 1px solid var(--border);
}

.dark .gradio-container {
    background: #1a3221;
    border-color: #2d4d38;
}

.upload-box {
    border: 2px dashed var(--border) !important;
    background: #f8faf9 !important;
    border-radius: 1rem !important;
    padding: 2rem !important;
}

.upload-box:hover {
    border-color: var(--primary) !important;
    background: #f0faf1 !important;
}

button {
    background: var(--primary) !important;
    color: white !important;
    padding: 1rem 2.5rem !important;
    border-radius: 0.875rem !important;
    transition: all 0.2s !important;
    font-weight: 600 !important;
}

button:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-2px);
}

.output-label {
    background: #e8f5e9 !important;
    border: 2px solid var(--border) !important;
    border-radius: 0.75rem !important;
    padding: 1.5rem !important;
    font-size: 1.1rem !important;
}

h1 {
    color: var(--primary) !important;
    text-align: center !important;
    margin-bottom: 1rem !important;
}

.description {
    color: #4a6350 !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
}
"""

# Build interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    <h1>🌿 Agroshield Plant Disease Detection</h1>
    <div class="description">
        Upload a plant leaf image for AI-powered disease diagnosis
    
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Leaf Image", elem_classes="upload-box")
            submit_btn = gr.Button("Analyze Image")
        
        with gr.Column():
            output_label = gr.Label(label="Diagnosis Results", elem_classes="output-label", num_top_classes=5)
    
    submit_btn.click(fn=predict, inputs=image_input, outputs=output_label)

demo.launch()