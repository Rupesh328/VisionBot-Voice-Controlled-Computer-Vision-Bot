import speech_recognition as sr
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import pyttsx3

recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Please say something...")
    engine = pyttsx3.init()
    engine.setProperty('rate', 140)

    engine.setProperty('volume', 1.0)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Female voice

    text = "please say something..., how can i help you..."
    engine.say(text)
    engine.runAndWait()
    audio = recognizer.listen(source)

    try:
        
        text = recognizer.recognize_google(audio)
        text = text.lower()
        print(text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        engine = pyttsx3.init()
        text = "Sorry, I could not understand the audio."
        engine.say(text)
        engine.runAndWait()
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
#please start the air Canvas
#please start the finger counter

if "air canvas" in text and "start" or "open" in text :
    engine = pyttsx3.init()
    text = "okay sir opening air canvas for you"
    engine.say(text)
    engine.runAndWait()
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    blue_index = 0
    green_index = 0
    red_index = 0
    yellow_index = 0

    
    kernel = np.ones((5, 5), np.uint8)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    
    paintWindow = np.zeros((471, 636, 3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8)
    mpDraw = mp.solutions.drawing_utils

    #  webcam
    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        # Read each frame from the webcam
        ret, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                   
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)
            print(center[1] - thumb[1])
            if (thumb[1] - center[1] < 30):
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            elif center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    paintWindow[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    colorIndex = 0  # Blue
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
            else:
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
        # Append the next deques when nothing is detected to avois messing up
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        # Draw lines of all the colors on the canvas and frame
        points = [bpoints, gpoints, rpoints, ypoints]
        
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)

        if cv2.waitKey(1) == ord('q'):
            engine = pyttsx3.init()
            text = "okay sir shutting down your canvas"
            engine.say(text)
            engine.runAndWait()
            break

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()
elif "finger counter" in text and "start" or "open" in text:
    engine = pyttsx3.init()
    text = "okay sir opening finger counter for you"
    engine.say(text)
    engine.runAndWait()
    # All the imports go here
    import cv2
    import mediapipe as mp

    # Initialize MediaPipe Hand model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Configure the Hand model
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    # Open the webcam
    cap = cv2.VideoCapture(0)


    # Function to count raised fingers based on hand landmarks
    def count_fingers(landmarks):
        # Define finger tip landmarks based on MediaPipe documentation
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        thumb_tip = 4  # Thumb tip

        fingers = []

        # Check for each finger if it's raised or not
        for tip in finger_tips:
            # Tip is higher than the joint (folded if higher Y value)
            if landmarks[tip].y < landmarks[tip - 2].y:
                fingers.append(1)  # Finger is raised
            else:
                fingers.append(0)  # Finger is not raised

        # Check thumb separately (based on x-coordinates, since thumb is horizontal)
        if landmarks[thumb_tip].x < landmarks[thumb_tip - 2].x:
            fingers.append(1)  # Thumb is raised
        else:
            fingers.append(0)  # Thumb is not raised

        return fingers.count(1)  # Return the count of raised fingers


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count the number of raised fingers
                finger_count = count_fingers(hand_landmarks.landmark)

                # Display the number of fingers raised on the screen
                cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Finger Counter', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            engine = pyttsx3.init()
            text = "okay sir shutting down your finger counter"
            engine.say(text)
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
elif  "face recognition" in text and "start" in text :
    engine = pyttsx3.init()
    text = "okay sir opening face recognition for you"
    engine.say(text)
    engine.runAndWait()
    
    import cv2
    import mediapipe as mp

    #  MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # the face detection model
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the image from BGR to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_frame)

        # Draw face landmarks and bounding boxes
        if results.detections:
            for detection in results.detections:
                # Draw bounding box and face landmarks
                mp_drawing.draw_detection(frame, detection)

                # Get the bounding box and score (confidence)
                bboxC = detection.location_data.relative_bounding_box
                score = detection.score[0]
                h, w, _ = frame.shape
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(
                    bboxC.height * h)

                # Display confidence score
                cv2.putText(frame, f'Score: {int(score * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        # Display the output frame
        cv2.imshow("Face Detection", frame)

        # Break the loop on pressing 'q'
        #you can use any letter for it
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
elif  "what's your name" in text :
    engine = pyttsx3.init()
    text = "Hello my name is luna ,what is your name?"
    engine.say(text)
    engine.runAndWait()

else :
    engine = pyttsx3.init()
    text = "can't recognise your voice please start me again"
    engine.say(text)

