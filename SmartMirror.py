import cv2
import mediapipe as mp
import os
import time
import serial
from PIL import Image
from sound import Sound
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')



faceDetector = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

portNo = "COM3"                         # указываем последовательный порт, к которому подключена Arduino
uart = serial.Serial(portNo, 9600)

isOpenImg = 0
isOpenImg1 = 0




p = [0 for i in range(21)]              # создаем массив из 21 ячейки для хранения высоты каждой точки
finger = [0 for i in range(5)]          # создаем массив из 5 ячеек для хранения положения каждого пальца


def distance(point1, point2):
  return abs(point1 - point2)


# For webcam input:
cap = cv2.VideoCapture(0)

with faceDetector.FaceDetection(min_detection_confidence=0.5) as face_detection, mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:

  while cap.isOpened():

    success, image = cap.read()

    start = time.time()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_detection.process(image)
    results0 = hands.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
      for id, detection in enumerate(results.detections):
        drawing.draw_detection(image, detection)

    #emo_detector = FER(mtcnn=True)
    #captured_emotions = emo_detector.detect_emotions(image)
    #print(captured_emotions)
    #dominant_emotion, emotion_score = emo_detector.top_emotion(image)
    #print(dominant_emotion, emotion_score)

    if results0.multi_hand_landmarks:
      for hand_landmarks in results0.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        for id, point in enumerate(hand_landmarks.landmark):
          # получаем размеры изображения с камеры и масштабируем
          width, height, color = image.shape
          width, height = int(point.x * height), int(point.y * width)

          p[id] = height  # заполняем массив высотой каждой точки

    distanceGood = distance(p[0], p[5]) + (distance(p[0], p[5]) / 2)
    # заполняем массив 1 (палец поднят) или 0 (палец сжат)
    finger[1] = 1 if distance(p[0], p[8]) > distanceGood else 0
    finger[2] = 1 if distance(p[0], p[12]) > distanceGood else 0
    finger[3] = 1 if distance(p[0], p[16]) > distanceGood else 0
    finger[4] = 1 if distance(p[0], p[20]) > distanceGood else 0
    finger[0] = 1 if distance(p[4], p[17]) > distanceGood else 0



    msg = ''
    inwork = Image.open(r"work.jpg")
    notwork = Image.open(r"unwork.jpg")
    if not (finger[0]) and finger[1] and not (finger[2]) and not (finger[3]) and finger[4]:
      msg = '@'
      if (isOpenImg1 == 0):
        Sound.volume_set(0)
        notwork.show()
        isOpenImg = 0
        isOpenImg1 =1
      elif (isOpenImg1 > 0):
        print("Изображение открыто")
    if not (finger[0]) and finger[1] and finger[2] and not (finger[3]) and not (finger[4]):
      msg = '$' + str(width) + ';'
    if (finger[0]) and not finger[1] and not (finger[2]) and not (finger[3]) and not finger[4]:
      msg = '*'+ str(width) + ';'
      if (isOpenImg == 0):
        Sound.volume_set(100)
        inwork.show()
        isOpenImg = 1
        isOpenImg1 = 0
        
      elif (isOpenImg > 0):
        print("Изображение открыто")







    # отправляем сообщение в Arduino
    if msg != '':
      msg = bytes(str(msg), 'utf-8')
      uart.write(msg)
      print(msg)


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    cv2.imshow('SmartMirror', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()

#python3 /Users/defolz/python-opencv/hack/SmartMirror.py
