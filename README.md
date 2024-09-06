# Person Detection and Tracking for Autism Spectrum Disorder Analysis

This project aims to detect and track multiple persons (children and therapists) in a video, assigning unique IDs, handling re-entries, post-occlusion tracking, and providing insights into behaviors, emotions, and engagement levels. The output includes annotated video frames with detected persons, along with their unique IDs and total number of persons in the video.

## Project Structure

- `tracking.py`: Main script for running the object detection and tracking.
- `test_videos`: the testing videos in the same project folder.
- `requirements.txt`: List of Python dependencies needed to run the project.
- `README.md`: Detailed instructions for setting up and running the project.

source_code/

    ├── dnn_model/
    
    │   ├── yolov4.weights (https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)
    
    │   ├── yolov4.cfg   (https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
    
    │   └── classes.txt  (renamed coco.names file)
    |
    ├── videos.mp4(in mp4 format)
    
    ├── tracking.py   =>Main script for running the object detection and tracking
    
    ├── requirements.txt
    
		-opencv-python==4.10.0
  
		-numpy==1.24.3
  
		-scipy==1.11.2
  
		-scikit-learn
  
		-opencv-python-headless
  
		-yolo4
  
		-opencv-python numpy
  


    └── README.md

## Installation

-->pip install -r requirements.txt 

### Prerequisites

1. **Python 3.8 or above**: Ensure that Python is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **pip**: Make sure you have pip installed for managing Python packages.



3. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
=======
# person_tracking



yolo weights(https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)

Drive link(https://drive.google.com/drive/folders/1z9S8UUHHmQpKiQxqw5hJTOLZ14lZEovT)
