import cv2
import mediapipe as mp
import numpy as np
import time
import array as arr

#creating variables from imported libaries
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

#video
cap = cv2.VideoCapture(0)

#variables
counter =0
stage = True
lapse = 0
i=0
cont=0
countdown = 5
down =0
Arr=arr.array('i',[])
startStage = True

Done = int(input("Enter How Long: "))
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        if startStage:
            startStage = False
            start=time.time()
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA
                        )
            # rep counter
            if down >= 0:
                now = time.time()
                down = countdown - (now - start)
                cv2.putText(image, f'{int(down)}', (170, 280), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 5)
            elif down < 0:
                if angle > 160:
                    if stage:
                        stage = False
                        counter += 1
                elif angle < 90:
                    if stage == False:
                        stage = True

                Time = time.time()
                TimeLeft = Done - (Time - start) + countdown
                cv2.putText(image, f'Counter: {int(counter)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 50, 50), 5)
                cv2.putText(image, f'Time left: {int(TimeLeft)}', (20, 450), cv2.FONT_HERSHEY_PLAIN, 3, (255, 50, 50), 5)
            if TimeLeft<0:
                print("Reps: ",counter)
                break
        except:
            pass

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()