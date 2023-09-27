import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os

face_x, face_y, face_w, face_h = 0, 0, 0, 0
input_path = '/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/LiveFeedFrames/1.jpg'
output_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/Images/afterProcessing1.jpg"
output_path2 = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/Images/afterProcessing2.jpg"


def crop_to_face(input_path, output_path):
    
    # Read input image in grayscale 
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) 
    print(f'IN CROP: {input_path}\n')
    
    assert img is not None, "file could not be read, check with os.path.exists()"
    
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
    #img_cropped = img[y1:y2, x1:x2]
    x1 = face_x 
    x2 = x1+face_w
    y1 = face_y 
    y2 = y1+face_h
    
    print(f'{x1} {x2} {y1} {y2}')
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

def numpy_equalization(input_path, output_path) :
    
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    img2 = cdf[img]
    
    cv2.imwrite(output_path, img2)
    
    return output_path

def opencv_equalization(input_path, output_path) :
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_equ = cv2.equalizeHist(img)
    cv2.imwrite(output_path, img_equ)
    return output_path

def process_images(input_file, user):
    # Open the file of image paths 
    read_file = open(input_file, 'r')
    # Open the file to write the preprocessed image paths to
    write_file = open(f'./outputFiles/{user}ProcessedImagePaths.txt', "w")
    
    
    count = 0; 
    while True: 
        
        img_paths = read_file.readlines()
        img_paths = [path.strip() for path in img_paths]
        
        for path in img_paths: 
            if os.path.exists(path): 
                
                print(path)
                print(path[16:])
                print(path[:-5])
                
                o_path = path[17:]
                o_path = o_path[:-4]
                img_out_path = f'./ProcessedFeedFrames/{o_path}Preprocessed.jpg'
                print(img_out_path)
              
                
    #     # Read file line by line
    #     lineIn = read_file.readline()
        
    #     # If reached end of file, break loop
    #     if not lineIn: 
    #         break 
    #     if count > 1 :
    #         break
 
    #     # Set input image path 
    #     img_path = [path.strip() for path in lineIn]
    #     print(f'{img_path}')
    #    # img_path = './LiveFeedFrames/0test6.jpg'
        
    #     # Set image output path --> saved to folder ProcessedFeedFrames
    #     img_out_path = f'./ProcessedFeedFrames{lineIn[16:]}{lineIn[:-5]}PreProcessed.jpg'
    #     print(img_out_path)
        
        # Preprocess Image
                crop_to_face(path, img_out_path)
                resize_image(path, img_out_path, 3.0)
                blur_image(path, img_out_path)
                numpy_equalization(path, img_out_path)
        
        # Write the path to the preprocessed image in the write file 
                write_file.write(f'{img_out_path}\n')
                count +=1
        break
    print("EOF")

    
process_images('/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/outputFiles/0testImagePaths.txt', '0test')
# img_path = crop_to_face(input_path, output_path)  
# img_path = resize_image(img_path, output_path, 3.0)
# img_path = blur_image(img_path, output_path)

# img_np_equ_path = numpy_equalization(img_path, output_path)
# img_cv_equ_path = opencv_equalization(img_path, output_path2)

# img_npequ = cv2.imread(img_np_equ_path, cv2.IMREAD_GRAYSCALE)
# img_cvequ = cv2.imread(img_cv_equ_path, cv2.IMREAD_GRAYSCALE)

# titles = ['Numpy Histogram Equalization', 'OpenCV Histogram Equalozation']
# images = [img_npequ, img_cvequ]

# for x in range(2): 
#     plt.subplot(1, 2, x + 1)
#     plt.imshow(images[x], cmap='gray')
#     plt.title(titles[x])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

