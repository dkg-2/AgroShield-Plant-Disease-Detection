import gradio as gr
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('plant_disease_model.h5')

# List of classes
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Define the prediction function
def predict(image):
    image = image.resize((224, 224))  # Resize to model's expected size
    image = np.array(image) / 255.0    # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return f"{predicted_class} ({confidence*100:.2f}%)"

# Build Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="🌿 Agroshield-Plant Disease Detection",
    description="Upload a plant leaf image and get the predicted disease class. Model trained on PlantVillage dataset."
)

# Launch app
interface.launch()

####################################################################################################

# import gradio as gr
# import tensorflow as tf
# import numpy as np
# import json

# # Load model and disease data
# model = tf.keras.models.load_model('plant_disease_model.h5')
# with open('diseases.json') as f:
#     disease_db = json.load(f)

# # Class names
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

# # Custom CSS (keep existing theme)
# custom_css = """
# :root {
#     --primary: #2e7d32;
#     --primary-hover: #1b5e20;
#     --surface: #f5fbf6;
#     --border: #c8e6c9;
# }

# /* ... keep all existing CSS rules ... */

# .disease-details {
#     background: #f0faf1 !important;
#     border: 2px solid var(--border) !important;
#     border-radius: 0.75rem !important;
#     padding: 1.5rem !important;
#     margin-top: 1.5rem;
# }

# .disease-details h3 {
#     color: var(--primary) !important;
#     margin-bottom: 0.75rem !important;
# }

# .disease-details p {
#     color: #4a6350 !important;
#     line-height: 1.6;
# }
# """

# def get_disease_info(disease_class):
#     """Retrieve formatted disease details"""
#     info = disease_db.get(disease_class, {
#         "cause": "Unknown pathogen",
#         "symptoms": "No specific symptoms identified",
#         "precaution": "Consult agricultural extension services",
#         "prevention": "Implement general crop protection measures",
#         "type": "Unknown",
#         "severity": "Undetermined"
#     })
    
#     return f"""
#     ### 🛡️ Disease Management Guide
    
#     **Pathogen Type**: {info['type']}  
#     **Risk Level**: {info['severity']}  
    
#     #### 🔍 Primary Causes
#     {info['cause']}
    
#     #### 🚨 Key Symptoms
#     {info['symptoms']}
    
#     #### ⚠️ Immediate Actions
#     {info['precaution']}
    
#     #### 🛡️ Long-term Prevention
#     {info['prevention']}
#     """

# def predict(image):
#     # Image processing
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
    
#     # Prediction
#     predictions = model.predict(image)
#     predicted_class = class_names[np.argmax(predictions)]
#     confidence = np.max(predictions)
    
#     return {
#         "diagnosis": f"{predicted_class} ({confidence*100:.2f}%)",
#         "details": get_disease_info(predicted_class)
#     }

# # Interface
# with gr.Blocks(css=custom_css) as demo:
#     gr.Markdown("""
#     <h1>🌿 Agroshield Plant Disease Detection</h1>
#     <div class="description">
#         AI-powered diagnosis with integrated crop protection guidance
    
#     """)
    
#     with gr.Row():
#         with gr.Column():
#             image_input = gr.Image(type="pil", label="📸 Leaf Image", elem_classes="upload-box")
#             submit_btn = gr.Button("🔍 Analyze Image", variant="primary")
        
#         with gr.Column():
#             output_label = gr.Label(label="🏥 Top Diagnosis", elem_classes="output-label", num_top_classes=5)
#             disease_details = gr.Markdown(elem_classes="disease-details")

#     submit_btn.click(
#         fn=predict,
#         inputs=image_input,
#         outputs=[output_label, disease_details]
#     )

# demo.launch()

##################################################


# import gradio as gr
# import tensorflow as tf
# import numpy as np
# import json

# # Load assets
# model = tf.keras.models.load_model('plant_disease_model.h5')
# with open('diseases.json') as f:
#     disease_db = json.load(f)


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

# def predict(image):
#     # Original prediction logic
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     predictions = model.predict(image)
#     predicted_class = class_names[np.argmax(predictions)]
#     confidence = np.max(predictions)
    
#     # Get JSON data
#     disease_info = disease_db.get(predicted_class, {})
    
#     # Format output
#     output = {
#         "Diagnosis": f"{predicted_class} ({confidence*100:.2f}%)",
#         "Type": disease_info.get("type", "Unknown"),
#         "Severity": disease_info.get("severity", "Undetermined"),
#         "Recommended Actions": [
#             disease_info.get("precaution", "Consult agricultural expert"),
#             disease_info.get("prevention", "Implement crop management practices")
#         ]
#     }
    
#     return output

# # Update interface
# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil"),
#     outputs=gr.JSON(label="Diagnosis Report"),
#     title="🌿 Agroshield-Plant Disease Detection",
#     description="Upload leaf image for diagnosis and crop protection guidance",
#     examples=["test_images/apple_scab.jpg", "test_images/tomato_blight.jpg"]
# )

# interface.launch()


