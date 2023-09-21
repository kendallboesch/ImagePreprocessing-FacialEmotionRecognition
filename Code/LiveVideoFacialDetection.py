# WRITTEN BY SAFWAN 
# Import the required modules

import cv2
import sys
import os
import time

def start_timer() :
    # Start the timer
    return time.time()

def stop_timer(t_start) : 
    # stop the timer 
    t_end = time.time()
    
    # Calculate time elapsed
    t_elapsed = t_end - t_start
    
    return t_elapsed

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# Load the Haar cascade model for face detection
face_cascade = cv2.CascadeClassifier(haar_model)

# Get the video source from the webcam
video_capture = cv2.VideoCapture(0)

#start timer 
t_start = start_timer()

# Loop until the user presses 'q' to quit
while True:
    

    # Capture a frame from the video source
    ret, frame = video_capture.read()

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the Haar cascade model
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around each face in the frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting frame with the face detection
    cv2.imshow('Video', frame)

    # Check if the user presses 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        t_elapsed = stop_timer(t_start)
        print(f"Running time: {t_elapsed}\n")
        break

# Release the video capture object and close the window
video_capture.release()

cv2.destroyAllWindows()