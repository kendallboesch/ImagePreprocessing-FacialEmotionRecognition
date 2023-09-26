import cv2 
import numpy as np 
import matplotlib.pyplot as plt

face_x, face_y, face_w, face_h = 0, 0, 0, 0
input_path = '/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/LiveFeedFrames/1.jpg'
output_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/Images/afterProcessing.jpg"


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
        
        # Save face measurements to global variables for future use 
        global face_x
        face_x = x
        global face_y
        face_y = y
        global face_h
        face_h = h
        global face_w
        face_w = w
        
    # Crop the image 
    img_cropped = img[y1:y2, x1:x2]
    
    # Save the cropped image
    cv2.imwrite(output_path, img_cropped)
    
    # Return the path to the newly cropped image 
    return output_path

def resize_image(input_path, output_path, scale_factor) :
    # Read image from path in grayscale 
    # No need for grayscale read in for how im
    # going to implement, but just for more modularity 
    img = cv2.imread(input_path)
    
    # Resize image dimensions 
    new_height = int(face_h * scale_factor)
    new_width = int(face_w * scale_factor) 
    
    # Resize image 
    img_resized = cv2.resize(img, (new_width, new_height))
    
    # Save resized image 
    cv2.imwrite(output_path, img_resized)
    
    # Return file path to resized image 
    return output_path

def blur_image(input_path, output_path) : 
    # Read in image in grayscale 
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Blur the image 
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Save blurred image 
    cv2.imwrite(output_path, img_blurred)
    
    # Return path to blurred image 
    return output_path

def histogram_equalization1(input_path, output_path) :  
    # Read the image in grayscale 
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    hist,bins = np.histogram(img.flatten(), 256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max())
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 225 / (cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

def numpy_equalization(input_path, output_path) :
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
        # Convert to gray scale 
   # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert img is not None, "file could not be read, check with os.path.exists()"
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    img2 = cdf[img]
    
    plt.imshow(img2, cmap ='gray')
    plt.show()

  
img_path = crop_to_face(input_path, output_path)  
img_path = resize_image(img_path, output_path, 3.0)
img_path = blur_image(img_path, output_path)
numpy_equalization(img_path, output_path)