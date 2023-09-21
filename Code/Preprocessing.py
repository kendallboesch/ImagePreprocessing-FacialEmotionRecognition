import cv2 
import numpy as np
import matplotlib.pyplot as plt

input_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/Images/testImg1.JPG"
output_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/afterProcessing.jpg"

face_x, face_y, face_w, face_h = 0, 0, 0, 0

def locate_face(image_path) :
    # image read 
    img = cv2.imread(image_path)
    
    # Convert to gray scale 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # load pretrained Haar Cascade Classifier 
    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    # perform facial detection 
    face = face_classifier.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40)
    )
    
    # draw bounding box 
    for(x, y, w, h) in face : 
       # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        
        # Set coordiantes for face location 
        global face_x
        face_x = x
        global face_y
        face_y = y
        global face_w
        face_w = w
        global face_h
        face_h = h
        
    # Print facial dimensions
    print(f"Face dimensions: \n\tX start: {face_x} \n\tY start: {face_y} \n\tHeight: {face_h} \n\tWidth: {face_w}") 
def crop_to_face(img_path_in) :
    # image read 
    img = cv2.imread(img_path_in)
    
    # Calculate dimensions 
    x1 = face_x 
    x2 = face_x + face_w
    y1 = face_y 
    y2 = face_y + face_h
    
    # crop image 
    cropped_image = img[y1:y2, x1:x2]
    
    # save cropped image 
    cv2.imwrite("cropped1.jpg", cropped_image)
    
    # return path to cropped image 
    #return cropped_image
    return "cropped1.jpg"
def resize_image(img_path_in, img_path_out, scale_factor) :
    # image read 
    img = cv2.imread(img_path_in)
    
    # resize dimensions 
    new_height = int(face_h * scale_factor) 
    new_width = int(face_w * scale_factor)
    
    # resize image 
    resized_image = cv2.resize(img, (new_width, new_height))
    
    # saved resized image 
    cv2.imwrite(img_path_out, resized_image)

    # return path to resized image 
    return resized_image 
def normalize_pixels(img_path_in) :
    # read image 
    img = cv2.imread(img_path_in)
    
    # convert image to float32 for precise calculations
    img_float = img.astype(np.float32)
    
    # normalize pixel values to range [0, 1] 
    img_normalized = img_float/255.0
    
    # return normalized np array of pizels
    return img_normalized

locate_face(input_path)
# print(f" X: {face_x} \t Y: {face_y} \t H: {face_h} \t W: {face_w}")

img_cropped = crop_to_face(input_path)


img_resized = resize_image(img_cropped, output_path, 3.0)

plt.imshow(img_resized)
plt.title("Post Processing")
plt.show()
    

    
     
    
        