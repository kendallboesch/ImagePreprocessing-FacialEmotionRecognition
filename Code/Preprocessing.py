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
    return img_path_out 
def normalize_pixels(img_path_in) :
    # read image 
    img = cv2.imread(img_path_in)
    
    # convert image to float32 for precise calculations
    img_float = img.astype(np.float32)
    
    # normalize pixel values to range [0, 1] 
    img_normalized = img_float/255.0
    
    # return normalized np array of pizels
    return img_normalized
def simple_thresholding(img_path): 
    #read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    #ensure image exists 
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # generate thresholded images & save
        # (a) Binary
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    cv2.imwrite("./Images/thresh_bianry.jpg", thresh1)
        # (b) Binary inverted 
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    cv2.imwrite("./Images/thresh_binary_inverted.jpg", thresh2)
        # (c) Truncate
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    cv2.imwrite("./Images/thresh_truncate.jpg", thresh3)
        # (d) To zero
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    cv2.imwrite("./Images/thresh_zero.jpg", thresh4)
        # (e) To zero, inverted
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    cv2.imwrite("./Images/thresh_zero_inverted.jpg", thresh5)
    
    # Assign titles for image show 
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    # Use matplotlib.pyplot library to display all thresholded images
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    print("Close the window, then press any key to close the window")
    plt.show()
    
  
    plt.waitforbuttonpress()
    
    plt.close()
    
    # return array of thresholded images 
    return images
def adaptive_thresholding(img_path) : 
    # Read in image from file path 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Make sure image exists 
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Apply Gaussian blur 
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding & save images
        # (a) Adaptive mean thresholding 
    th1 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("./Images/aThresh_mean.jpg", th1)
        # (b) Adaptive Gaussian thresholding 
    th2 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("./Images/aThresh_Gaussian.jpg", th2)
    
    # Assign titles for image show
    titles = ['Original Image', 'Blurred Image', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # Create array of images
    images = [img, img_blurred, th1, th2]
    
    # Use matplotlib.pyplot library to display images 
    for i in range(4) :
        plt.subplot(2,2,i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

    
    # Return image array
    return images

    
    
locate_face(input_path)
# print(f" X: {face_x} \t Y: {face_y} \t H: {face_h} \t W: {face_w}")

img_cropped = crop_to_face(input_path)


img_resized = resize_image(img_cropped, output_path, 3.0)

simple_thresholding(img_resized)

adaptive_thresholding(img_resized)

# plt.imshow(img_resized)
# plt.title("Post Processing")
# plt.show()
    

    
     
    
        