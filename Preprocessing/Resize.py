import cv2
import numpy as np

def resize_image(image_path, output_path, width, height) : 
    try: 
        # load image 
        img = cv2.imread(image_path) 
        
        #resize 
        resized_img = cv2.resize(img, (width, height))
        
        # save resized image 
        cv2.imwrite(output_path, resized_img)
        
        print("image resized successfully!")
        
    except Exception as e:
        print(f"an error occured: {e}")
        
def crop_image(image_path, output_path) :
    try : 
        img = cv2.imread(image_path)
        
        #print image shape 
        print(img.shape)
        cv2.imshow("original", img)
        
        #crop 
        cropped_image = img[80:280, 150:330]
        
        #display cropped image 
        cv2.imshow("cropped", cropped_image)
        
        #save cropped image 
        cv2.imwrite("cropped_img.jpg", cropped_image)   
        
        print("Image cropped successfully")
    except Exception as e : 
        print(f"An error occured: {e}")
        
if __name__ == "__main__":
    input_image_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/testImg1.JPG"  # Change this to your image path
    output_image_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/output_resized_image.jpg"  # Change this to your desired output path
    target_width = 400  # Change this to your desired width
    target_height = 400  # Change this to your desired height

    #resize_image(input_image_path, output_image_path, target_width, target_height)
    
    cropped_image_out = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/cropped_image.jpg"
    crop_image(input_image_path, cropped_image_out)
    
    