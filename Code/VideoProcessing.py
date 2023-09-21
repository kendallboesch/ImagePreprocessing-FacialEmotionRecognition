import csv 
import cv2 
import numpy as np
from pytube import YouTube



output_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/availableVideos.csv"
accessed = 0 
processed = 0

with open(output_path, 'w') as out :
    out.write("link,start,end,video,utterance,arousal,valence,EmotionMaxVote")


def process_video(url, t_start, t_end, video, utterance, arousal, valence, emotionMaxVote, validCount, numProcessed) : 
    
    #initialize video capture 
    vid = cv2.VideoCapture(url)
    numProcessed += 1
    
    if numProcessed % 10 == 0 : 
        print(f"\t\t\t\t NUM PROCESSED: {numProcessed}")
    
    #ensure video opened 
    if not vid.isOpened(): 
        print("Error") 
    else :
        print("OPENED")
        validCount +=1
        entry = url + ',' + str(t_start) + ',' + str(t_end) + ',' + video + ',' + utterance + ',' + str(arousal)+','+str(valence) + ',' + str(emotionMaxVote) + '\n'
        with open(output_path, 'w') as out :
            out.write(entry)
            

def download_youtube(youtube_url, num) :
    
    #create youtube object
    yt = YouTube(youtube_url)
    
   
    
    # download the video 
    try :
        # select highest resolution 
        stream = yt.streams.get_highest_resolution()
    
        # create video file string 
        file = "video" + str(num) + ".mp4"
        stream.download(output_path = ".", filename=file)  
        video_path = file
    
    except Exception as e : 
        print(f"ERROR : {e}")
        video_path = "" 
        
    return video_path
        
    
    


    
# csv_file_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/omg_TrainVideos.csv"
csv_file_path = "/Users/kendallboesch/Desktop/CS5351-SeniorDesign/TestCQ/SampleTraining.csv"

    # Open & read training dataset csv 
with open(csv_file_path, 'r') as csvfile :
    num_vid = 0
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader : 
        url = row['link']
        t_start = float(row['start'])
        t_end = float(row['end'])
        video = row['video']
        utterance = row['utterance']
        arousal = float(row['arousal'])
        valence = float(row['valence'])
        emotionMaxVote = int(row['EmotionMaxVote'])

            # Process video 
        converted_file_path = download_youtube(url, num_vid)    
        num_vid += 1
        process_video(converted_file_path, t_start, t_end, video, utterance, arousal, valence, emotionMaxVote, accessed, processed)