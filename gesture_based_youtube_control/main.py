import os
# Suppress TensorFlow Lite warnings from MediaPipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
# Suppress absl (TensorFlow) logging warnings
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

import warnings
warnings.filterwarnings('ignore')

from models.model_architecture import model
import pandas as pd
import mediapipe as mp
import numpy as np
from utils import *

################################################### VARIABLES INITIALIZATION ###########################################################

# Set to normal mode (=> no recording of data)
mode = 0
CSV_PATH = 'data/gestures.csv'

# Camera settings
WIDTH = 1028//2
HEIGHT = 720//2

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)


# important keypoints (wrist + tips coordinates)
# for training the model
TRAINING_KEYPOINTS = [keypoint for keypoint in range(0, 21, 4)]


# Mouse mouvement stabilization
SMOOTH_FACTOR = 6
PLOCX, PLOCY = 0, 0 # previous x, y locations
CLOX, CLOXY = 0, 0 # current x, y locations


# Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75)

# Hand landmarks drawing
mp_drawing = mp.solutions.drawing_utils

# Load saved model for hand gesture recognition
GESTURE_RECOGNIZER_PATH = 'models/model.pth'
model.load_state_dict(torch.load(GESTURE_RECOGNIZER_PATH))

# Load Label
LABEL_PATH = 'data/label.csv'
labels = pd.read_csv(LABEL_PATH, header=None).values.flatten().tolist()


# confidence threshold(required to translate gestures into commands)
CONF_THRESH = 0.7  # Lowered from 0.9 to help with Backward gesture detection

# history to track the n last detected commands
GESTURE_HISTORY = deque([])

# general counter (for volum up/down; forward/backward)
GEN_COUNTER = 0


################################################### INITIALIZATION END ###########################################################



while True:
    key = cv.waitKey(1) 
    if key == ord('q'):
        break

    
    # choose mode (normal or recording)
    mode = select_mode(key, mode=mode)

    # class id for recording
    class_id = get_class_id(key)

    # read camera
    has_frame, frame = cap.read()
    if not has_frame:
        break

    # horizontal flip and color conversion for mediapipe
    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Mouse zone for mouse movement (covers whole frame)
    frame_height, frame_width = frame.shape[:2]
    m_zone = np.array([(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)])

############################################ GESTURE DETECTION / TRAINING POINT LOGGING ###########################################################
 
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get hand handedness (left or right)
            handedness = results.multi_handedness[idx]
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            confidence = handedness.classification[0].score

            # draw landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Draw hand label on the frame
            hand_center = hand_landmarks.landmark[9]  # Middle finger MCP
            x = int(hand_center.x * frame_width)
            y = int(hand_center.y * frame_height)
            
            # Color code: Green for Right hand, Blue for Left hand
            color = (0, 255, 0) if hand_label == 'Right' else (255, 0, 0)
            cv.putText(frame, f'{hand_label} Hand', (x - 50, y - 20),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)

            # get landmarks coordinates
            coordinates_list = calc_landmark_coordinates(frame_rgb, hand_landmarks)
            important_points = [coordinates_list[i] for i in TRAINING_KEYPOINTS]


            # Conversion to relative coordinates and normalized coordinates
            preprocessed = pre_process_landmark(important_points)
            
            # compute the needed distances to add to coordinate features
            d0 = calc_distance(coordinates_list[0], coordinates_list[5])
            pts_for_distances = [coordinates_list[i] for i in [4, 8, 12]]
            distances = normalize_distances(d0, get_all_distances(pts_for_distances)) 
            

            # Write to the csv file "keypoint.csv"(if mode == 1)
            # logging_csv(class_id, mode, preprocessed)
            features = np.concatenate([preprocessed, distances])
            draw_info(frame, mode, class_id)
            logging_csv(class_id, mode, features, CSV_PATH)

            
            # inference
            conf, pred = predict(features, model)
            gesture = labels[pred]

            
####################################################### YOUTUBE PLAYER CONTROL ###########################################################                       
                
            # check if prediction confidence is higher than a given threshold (gestures work anywhere in frame)
            if conf >= CONF_THRESH:
                
                # Print hand detection info for debugging
                print(f"Detected: {hand_label} Hand - Gesture: {gesture} - Confidence: {conf:.2f}") 

                # track command history
                gest_hist = track_history(GESTURE_HISTORY, gesture)

                if len(gest_hist) >= 2:
                    before_last = gest_hist[len(gest_hist) - 2]
                else:
                    before_last = gest_hist[0]
                
                # Debug gesture history
                if gesture == 'Backward':
                    print(f"DEBUG: Gesture history: {list(gest_hist)}, before_last: {before_last}")

            ############### mouse gestures ##################
                if gesture == 'Move_mouse':
                    # Convert hand position to screen coordinates (full frame)
                    screen_size = pyautogui.size()
                    screen_width, screen_height = screen_size
                    
                    # Get hand position (middle finger MCP)
                    hand_x, hand_y = coordinates_list[9]
                    
                    # Map hand position to screen coordinates
                    x = np.interp(hand_x, (0, frame_width), (0, screen_width))
                    y = np.interp(hand_y, (0, frame_height), (0, screen_height))
                    
                    # smoothe mouse movements
                    CLOX = PLOCX + (x - PLOCX) / SMOOTH_FACTOR
                    CLOXY = PLOCY + (y - PLOCY) / SMOOTH_FACTOR
                    pyautogui.moveTo(CLOX, CLOXY)
                    PLOCX, PLOCY = CLOX, CLOXY

                if gesture == 'Right_click' and before_last != 'Right_click':
                    pyautogui.rightClick()

                if gesture == 'Left_click' and before_last != 'Left_click':
                    pyautogui.click()    


            ############### Other gestures ################## 
                # Different functionality based on hand type
                if hand_label == 'Right':
                    # Right hand controls volume and playback
                    if gesture == 'Play_Pause' and before_last != 'Play_Pause':
                        pyautogui.press('space')
                        print("Right Hand: Play/Pause triggered")
                    
                    elif gesture == 'Vol_up_gen':
                        pyautogui.press('volumeup')
                        print("Right Hand: System Volume Up")

                    elif gesture == 'Vol_down_gen':
                        pyautogui.press('volumedown')
                        print("Right Hand: System Volume Down")

                    elif gesture == 'Vol_up_ytb':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 2 == 0:  # Reduced from 4 to 2 for more responsive control
                            # Try multiple methods to increase volume
                            print(f"Right Hand: Volume UP triggered! Counter: {GEN_COUNTER}")
                            pyautogui.press('up')  # YouTube volume up
                            pyautogui.press('volumeup')  # System volume up as backup

                    elif gesture == 'Vol_down_ytb':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 2 == 0:  # Reduced from 4 to 2 for more responsive control
                            # Try multiple methods to decrease volume
                            print(f"Right Hand: Volume DOWN triggered! Counter: {GEN_COUNTER}")
                            pyautogui.press('down')  # YouTube volume down
                            pyautogui.press('volumedown')  # System volume down as backup
                
                elif hand_label == 'Left':
                    # Left hand controls navigation and other functions
                    if gesture == 'Play_Pause' and before_last != 'Play_Pause':
                        pyautogui.press('space')
                        print("Left Hand: Play/Pause triggered")
                    
                    elif gesture == 'Forward':
                        GEN_COUNTER += 1
                        if GEN_COUNTER % 4 == 0:
                            pyautogui.press('right')
                            print("Left Hand: Forward triggered")
                    
                    elif gesture == 'Backward':
                        GEN_COUNTER += 1
                        print(f"DEBUG: Backward detected! Counter: {GEN_COUNTER}, GEN_COUNTER % 4 = {GEN_COUNTER % 4}")
                        # Make Backward more responsive - trigger every 2nd detection instead of 4th
                        if GEN_COUNTER % 2 == 0:
                            pyautogui.press('left')
                            print("Left Hand: Backward triggered")
                        else:
                            print(f"DEBUG: Backward gesture detected but not triggered (counter: {GEN_COUNTER})")
                    
                    elif gesture == 'fullscreen' and before_last != 'fullscreen':
                        pyautogui.press('f')
                        print("Left Hand: Fullscreen triggered")
                    
                    elif gesture == 'Cap_Subt' and before_last != 'Cap_Subt':
                        pyautogui.press('c')
                        print("Left Hand: Captions triggered")
                
                # Common gestures that work with both hands
                if gesture == 'Neutral':
                    GEN_COUNTER = 0 

                # show detected gesture with hand info
                cv.putText(frame, f'{hand_label} Hand: {gesture} | {conf: .2f}', (int(WIDTH*0.05), int(HEIGHT*0.07)),
                    cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)
                
                # Special debug for Backward gesture
                if gesture == 'Backward':
                    cv.putText(frame, f'BACKWARD DETECTED! Counter: {GEN_COUNTER}', (int(WIDTH*0.05), int(HEIGHT*0.15)),
                        cv.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)

                



    cv.imshow('', frame)
cap.release()
cv.destroyAllWindows()
