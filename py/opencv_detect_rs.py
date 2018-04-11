import pyrealsense2 as rs
from imutils import paths
import numpy as np
import imutils
import cv2

frontal_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface.xml')
profile_face_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')
# full_body_cascade = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')
hands_cascade = cv2.CascadeClassifier('data/haarcascade_hands.xml')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # Convert images to numpy arrays
    image = np.asanyarray(color_frame.get_data())
    # image = imutils.resize(image, width=min(800, image.shape[1]))

    faceColor = (0, 255, 0)
    # bodyColor = (255, 0, 0)
    # handColor = (255, 0, 0)

    # Detect frontal faces in the image and draw face rect
    faces = frontal_face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), faceColor, 2)

    # Detect profile faces in the image and draw face rect
    # faces = profile_face_cascade.detectMultiScale(image, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), faceColor, 2)

    # Detect profile faces in the image and draw face rect
    # Does not work right now
    '''
    body = full_body_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in body:
        cv2.rectangle(image, (x, y), (x + w, y + h), bodyColor, 2)
    '''

    # Detect profile faces in the image and draw face rect
    # Works from time to time
    # hands = hands_cascade.detectMultiScale(image, 1.3, 5)
    # for (x, y, w, h) in hands:
    #     cv2.rectangle(image, (x, y), (x + w, y + h), handColor, 2)

    cv2.imshow('Detection results', cv2.WINDOW_AUTOSIZE)

    # Press Q to exit the video playback
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()
