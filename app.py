from fastapi import FastAPI, File, UploadFile
import keras_ocr
import cv2
import numpy as np

app = FastAPI()

pipeline = keras_ocr.pipeline.Pipeline()

def process_image(image_bytes):
    # Convert the byte array into a NumPy array
    npimg = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Image decoding failed.")
    
    # Convert the image to RGB
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Recognize the text
    prediction_groups = pipeline.recognize([img_array])
    return prediction_groups

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    results = process_image(image_bytes)
    
    predicted_text = []
    for text, _ in results[0]:
        predicted_text.append(text)
    
    # Join the predicted text as a single string, ensuring the order is preserved
    return {"predicted_text": " ".join(predicted_text)}
