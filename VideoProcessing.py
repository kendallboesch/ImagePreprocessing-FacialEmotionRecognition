import csv 
import cv2 

def process_video(url, t_start, t_end) : 
    
    #initialize video capture 
    vid = cv2.VideoCapture(url)
    
    #ensure video opened 
    if not vid.isOpened(): 
        print("Error")
    else :
        print("OPENED")
        
    # try : 
    #     vid = cv2.VideoCapture(url)
    #     print(f"File Opened: {url}")
    
    # except Exception as e : 
    #     print(f"Error: Could not open video - {e}") 
    #     return 
    
csv_file_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/SampleTraining.csv"
    
    # Open & read training dataset csv 
with open(csv_file_path, 'r') as csvfile :
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader : 
        url = row['link']
        t_start = float(row['start'])
        t_end = float(row['end'])
            
            # Process video 
            
        process_video(url, t_start, t_end)