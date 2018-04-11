from imutils import paths
import numpy as np
import imutils
import cv2

frontal_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface.xml')
profile_face_cascade = cv2.CascadeClassifier('data/haarcascade_profileface.xml')
# full_body_cascade = cv2.CascadeClassifier('data/haarcascade_fullbody.xml')
hands_cascade = cv2.CascadeClassifier('data/haarcascade_hands.xml')

cam = cv2.VideoCapture(0)

while cam.isOpened():
    success, image = cam.read()

    if not success:
        continue

    image = cv2.flip(image, 1)
    image = imutils.resize(image, width=min(800, image.shape[1]))

    faceColor = (0, 255, 0)
    # bodyColor = (255, 0, 0)
    handColor = (255, 0, 0)

    # Detect frontal faces in the image and draw face rect
    faces = frontal_face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), faceColor, 2)

    # Detect profile faces in the image and draw face rect
    faces = profile_face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), faceColor, 2)

    # Detect profile faces in the image and draw face rect
    # Does not work right now
    '''
    body = full_body_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in body:
        cv2.rectangle(image, (x, y), (x + w, y + h), bodyColor, 2)
    '''

    # Detect profile faces in the image and draw face rect
    # Works from time to time
    hands = hands_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in hands:
        cv2.rectangle(image, (x, y), (x + w, y + h), handColor, 2)

    cv2.imshow('Detection results', image)

    # Press Q to exit the video playback
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()
cam.release()
