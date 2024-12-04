import cv2
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech

#pip install ultralytics ,opencv-python
# Function to give a voice command
def give_voice_command(command):
    engine.say(command)
    engine.runAndWait()


# Initialize the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Open the webcam
cap = cv2.VideoCapture(0)
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

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if a face is detected on the left side of the frame
    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        frame_center_x = frame.shape[1] // 2

        # If the face is on the left side of the frame
        if face_center_x < frame_center_x // 2:
            dd0 += 1

            if dd0 == 20:
                dd0 = 0
                give_voice_command("Turn right")
                break  # Avoid multiple commands in one frame

            # If the face is on the right side of the frame
        elif face_center_x > frame_center_x + frame_center_x // 2:
            dd1 += 1

            if dd1 == 20:
                dd1 = 0
                give_voice_command("Turn left")
                break  # Avoid multiple commands in one frame


        else:
            dd2 += 1

            if dd2 == 20:
                dd2 = 0
                give_voice_command("Obstacle ahead. Please be careful")
                break  # Avoid multiple commands in one frame

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Face Detection', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


