

def Modules(video_path,NUM_FRAME,SCALES):
    import cv2
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    count, position = 0, None

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break  # End of the video file

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imlist = []

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(result.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    X, Y = int(lm.x * w), int(lm.y * h)
                    imlist.append([id, X, Y])

            # Count push-ups by analyzing the positions
            if imlist:
                # Check the coordinates to decide the push-up state
                shoulder_r = imlist[12][2]  # Right shoulder
                shoulder_l = imlist[11][2]  # Left shoulder
                elbow_r = imlist[14][2]     # Right elbow
                elbow_l = imlist[13][2]     # Left elbow

                if (shoulder_r >= elbow_r and shoulder_l >= elbow_l):
                    if position != "down":
                        position = "down"
                elif (shoulder_r < elbow_r and shoulder_l < elbow_l):
                    if position == "down":
                        position = "up"
                        count += 1
            cv2.imshow("Gesture Counter", image)
            key = cv2.waitKey(1)
            if key == ord('i'):
                break
    return ("GESTURE COUNT: ",count)