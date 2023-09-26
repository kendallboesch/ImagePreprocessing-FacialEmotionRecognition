import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def crop_to_face(input_path, output_path):
    
    # Read input image in grayscale 
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 
    
    # Load pretrained Haar Cascade Classifier 
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Perform facial detection 
    face = face_classifier.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )
    
    # Get face coordinates 
    for(x, y, w, h) in face : 
        x1 = x
        x2 = x + w 
        y1 = y
        y2 = y + h
    
    # Crop the image 
    img_cropped = img[y1:y2, x1:x2]
    
    # Save the cropped image
    cv2.imwrite(output_path, img_cropped)
    
    # Return the path to the newly cropped image 
    return output_path
        
    