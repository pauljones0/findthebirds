# Load the yolov8 nano model
from ultralytics import YOLO
from speedster import optimize
model = YOLO("yolov8n.pt")
model = optimize(model)


# Define a function to get the timestamps from .avi files
def get_timestamps(file):
    # Use OpenCV to read the video file
    import cv2
    cap = cv2.VideoCapture(file)
    # Get the frame rate and number of frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize a list to store the timestamps
    timestamps = []
    # Loop through each frame and append its timestamp
    for i in range(n_frames):
        # Calculate the timestamp in seconds
        timestamp = i / fps
        # Append it to the list
        timestamps.append(timestamp)
    # Release the video capture object and return the list
    cap.release()
    return timestamps


# Define a function to find birds in .avi files and print out their start and end timestamps
def find_birds(file, output_dir):
    # Get the timestamps from the file using our previous function
    timestamps = get_timestamps(file)

    # Run detection on the file using our model and get back a list of results objects
    results = model(file)

    # Initialize a variable to store the previous frame's detections
    prev_detections = []

    # Initialize variables to store start and end times for each bird detection
    start_time = None
    end_time = None
    foundBird = False
    # Loop through each result object
    for result in results:
        # Get the current frame's detections as a list of tuples (class, confidence, bbox)
        curr_detections = result.boxes.xyxy[0].tolist()

        # Filter out only bird detections (class index is zero)
        curr_birds = [d for d in curr_detections if d[5] == "bird"]

        # Check if there are any bird detections in this frame
        if curr_birds:
            # If this is the first frame with bird detections, set start time as current timestamp
            if start_time is None:
                start_time = timestamps[result.frame]

                # Update end time as current timestamp
            end_time = timestamps[result.frame]

            # Save this frame's detections for comparison with next frame
            prev_detections = curr_birds
            foundBird = True
        else:
            # If there are no bird detections in this frame, check if there were any in previous frame
            if prev_detections:
                # If yes, then we have reached an end of a bird detection segment
                print(f"Bird detected from {start_time} s to {end_time} s")

                # Reset start time and end time variables
                start_time = None
                end_time = None

                # Clear previous detections list
                prev_detections.clear()

    if foundBird:        # Write output video file with bounding boxes drawn on detected birds
        result.save(output_dir)


import os
dirname = 'E:/DCIM/100MEDIA'
output_dir_path = "E:/DCIM/output"
for file in os.listdir(dirname):
    if file.endswith('.avi'):
        input_file_path = os.path.join(dirname, file)
        find_birds(input_file_path, output_dir_path)
