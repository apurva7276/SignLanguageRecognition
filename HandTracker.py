import cv2
import mediapipe as mp
import numpy as np
import os
from matplotlib import pyplot as plt
import time
from keras.models import load_model
import Required_variables as rv

# Path for exported data i.e numpy arrays
DATA_PATH = os.path.join('MP_Data3')

#model name
modelName = rv.get_modelName()

# Actions that we are try to detect
# actions = np.array(['hello', 'thanks', 'iloveyou'])
#actions = rv.get_actions()
actions = rv.detect_actions_in_folder()

# Thirty sequences worth of data for each action
no_of_sequences = 30

mpHands = mp.solutions.hands
# hands= mpHands.Hands(False, 2, 1, 0.8, 0.5);

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


#
def mediapipe_detection(image, model):
    # the mediapipe holistic object(model) requires the image to be in
    # RGB type and the CV2 object(cap) outputs a BGR type image
    # here we are converting the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# drawing the landmarks collected by mediapipe_detection()
# The color for landmarks is currently set to default(Red)
# Also the function does not return the image but applies the landmark drawing
# to the current image in place
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    # pose model will return landmarks but the visibilty value inside of each landmark will be low if the pose is not visible
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

    # face and hand landmarks will be empty if face and hands are not visible in the frame
    # Thats why we are using the else statement , if landmarks are empty we are getting
    # numpy arrays of same length but filled with zero's
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    flag1 = True
    res1 = not np.any(lh)
    res2 = not np.any(rh)
    if res1 and res2:
        return np.concatenate([pose, face, lh, rh]), flag1
    else:
        flag1 = False
    return np.concatenate([pose, face, lh, rh]), flag1



colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


def start_detection():
    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.5

    model = load_model(modelName)
    flag = False

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw landmarks
            #draw_landmarks(image, results)

            # 2. Prediction logic
            keypoints, flag = extract_keypoints(results)
            #         sequence.insert(0,keypoints)
            #         sequence = sequence[:30]
            # res1 = not np.any(keypoints)
            print("Hand Not Detected: ", flag)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            no_hand_msg = "Please show hands!!"
            predicted_text = ""
            predicted_word = ""
            # print(flag)
            if flag:
                predicted_word = no_hand_msg
            else:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    # print(actions[np.argmax(res)])
                    predicted_word = actions[np.argmax(res)]
                    # 3. Viz logic
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    # image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            # cv2.imshow('OpenCV Feed', image)

            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, " ", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
            cv2.putText(blackboard, "Predicted text- " + predicted_word, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 255, 0))
            cv2.putText(blackboard, predicted_word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))

            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res = np.hstack((image, blackboard))
            cv2.imshow("Recognizing gesture", res)

            keypress = cv2.waitKey(1)

            if keypress == ord('q') or keypress == ord('c'):
                break
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()


start_detection()