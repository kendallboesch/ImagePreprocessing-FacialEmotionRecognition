import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/Images/testImg1.JPG"
output_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/afterProcessing.jpg"
face_x = 0 
face_y = 0 
face_width = 0
face_height = 0

#image read 
img = cv2.imread(image_path)

#print dimensions 
print("original measures: ", img.shape) 
print('\n')

                    # FACIAL DETECTION 
#convert to grayscale 
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#get dimensions of grayscale image 
print("gray scale measures: ", gray_image.shape) 
print('\n')

# #load pre-trained Haar Cascase classifier 
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
#perform facial detection 
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)
#draw bounding box
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    face_x = x 
    face_y = y 
    face_width = w
    face_height = h; 

# print face dimension values 
print(f"Face dimensions: \n\tX start: {face_x} \n\tY start: {face_y} \n\tHeight: {face_height} \n\tWidth: {face_width}") 
    
#display image 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite("img_bgr.jpg", img)

cv2.imwrite(output_path, img_rgb)

                    # CROP TO FACE 
# get face box x & y vales
x1 = face_x 
x2 = face_x + face_width
y1 = face_y 
y2 = face_y + face_height

#crop the image 
cropped_image = img[y1:y2, x1:x2] 

# save cropped image 
cv2.imwrite('cropped_image.jpg', cropped_image)

                    # RESIZE CROPPED IMAGE
# set scale factor 
scale_factor = 3.0

# retrieve height & width 
height,width = cropped_image.shape[:2]
 
# calculate new heights 
new_height = int(height * scale_factor)
new_width = int(width * scale_factor)

# resize the cropped image
resized_cropped_image = cv2.resize(cropped_image, (new_width, new_height))
 
# save the resized image 
cv2.imwrite('resized_cropped_image.jpg', resized_cropped_image) 

                    # DATA NORMALIZATION
# convert image to float32 for precise calculations 
image_float = resized_cropped_image.astype(np.float32)

# normalize pixel valyes to range [0, 1]
normalized_image = image_float/255.0

# scale image 
scaled_image = (normalized_image * 255).astype(np.uint8)

#save normalized image 
cv2.imwrite('normalized_image.jpg', scaled_image)

cv2.imshow('Final', scaled_image)
 
                    # IMAGE BLURRING 

#blurred_image = cv2.GaussianBlur(resized_cropped_image,)
