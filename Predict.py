import cv2
from ultralytics import YOLO
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech


def give_voice_command(command):
    engine.say(command)
    engine.runAndWait()


model = YOLO('yolov8m.pt')  # Use YOLOv8n for better performance on smaller devices

# Define the class IDs for detection (Person, Bicycle, Car as examples)
class_ids = [0,1, 2,3,5]  # Person, Bicycle, Car

# Class names for COCO dataset
class_names = ['person','Bicycle', 'Car','motorcycle', 'bus']

# Open webcam (change to video path if using a video file)
cap = cv2.VideoCapture(0)  # Use 'video.mp4' or another file instead of 0 for videos
dd0 = 0
dd1 = 0
dd2 = 0
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform detection
    results = model(frame)

    # Filter detections for the specified class IDs
    for result in results:
        for box in result.boxes:
            if int(box.cls) in class_ids:  # Check if the detected class is in the list
                class_id = int(box.cls)
                label = class_names[class_ids.index(class_id)]  # Get class label
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf.item()  # Convert tensor to float

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Get frame dimensions
                #frame_width = frame.shape[1]

                frame_height, frame_width = frame.shape[:2]  # Ensure this gives correct dimensions
                frame_center_x = frame_width // 2

                bounding_box_center_x = (x1 + x2) // 2  # Center x-coordinate of the bounding box

                if 0 <= x1 <= 200:
                    dd0 += 1
                    if dd0 == 20:
                        dd0 = 0
                        give_voice_command("detected Object Name '" + label + "' on the Right side. Turn Left.")
                        print("Person is on the Right side of the frame.")

                if  300 <= x1 <= 500:
                    dd0 += 1
                    if dd0 == 20:
                        dd0 = 0
                        give_voice_command("detected Object Name '" + label + "'on the Left right. Turn Right.")
                        print("Person is on the Left side of the frame.")
                if 201 <= x1 <= 299:
                    dd2 += 1

                    if dd2 == 20:
                        dd2 = 0
                        give_voice_command("Obstacle ahead. Please be careful")
                        break


                #print(f"Frame Width: {frame_width}")
                #print(f"x1: {x1}, Frame Center X: {frame_center_x}, Bounding Box Center X: {bounding_box_center_x}")







    # Show the frame
    cv2.imshow('YOLOv8 Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

'''coco_class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]'''