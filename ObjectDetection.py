import sys
import cv2
import os
import numpy as np
from ObjectTracking import ObjectTracking

import tensorflow as tf
from src.get_pitch_frames import get_pitch_frames
from src.generate_overlay import generate_overlay


class ObjectDetection:
    def __init__(self):
        if os.path.isfile('static/model/HAAR_CASCADES/fullbody.xml'):
            self.detector = cv2.CascadeClassifier('static/model/HAAR_CASCADES/fullbody.xml')
        else:
            print('Haar cascade could not loaded!')
            self.detector = False
            sys.exit()

        if os.path.isfile('static/model/yolov4-tiny-baseball-416/saved_model.pb'):
            self.yolov4_weights = 'static/model/yolov4-tiny-baseball-416'

    def detect_image(self, image_path):

        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.detector:
            detections = self.detector.detectMultiScale(image_gray)
            print(detections)
            print(len(detections), "people detected!\n")

            for (x, y, width, height) in detections:
                print(x, y, width, height)
                cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 1)
                cv2.imshow('Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def detect_track_video(self, video_path):
        tr = ObjectTracking("CSRT", video_path)
        tracker = tr.track_with('CSRT')
        # print(tracker)
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print('Could not load video!')
            sys.exit()

        ret, frame = video.read()
        if not ret:
            print('Error loading the first frame!')
            sys.exit()

        while True:
            ret, frame = video.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(frame_gray)
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow('Detection-Tracking', frame)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if x > 0:
                    print('Detection Successful!')
                    return x, y, w, h

    def yolov4_ball_detection(self, video_path):
        # Initialize variables
        size = 416
        iou = 0.45
        score = 0.5
        pitch_frames = []
        output_path = 'static/overlay/overlay.avi'
        saved_model_loaded = tf.saved_model.load(self.yolov4_weights)
        infer = saved_model_loaded.signatures['serving_default']
        try:
            ball_frames, width, height, fps = get_pitch_frames(video_path, infer, size, iou, score)
            pitch_frames.append(ball_frames)
        except Exception as e:
            print(
                f'Error: Sorry we could not get enough baseball detection from the video, video {video_path} will not be overlayed')
            print(e)
        if len(pitch_frames):
            generate_overlay(pitch_frames, width, height, fps, output_path)

    def ball_detection(self, live=True, video_path=""):
        if live:
            video = cv2.VideoCapture(1)
        else:
            video = cv2.VideoCapture(video_path)
        # Find best circle which is the circle closest to the previous circle
        previousCircle = None  # The circle from the previous frame

        # small lambda function to calculate the distance between two points
        distance = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.GaussianBlur(frame_gray, (17, 17), 0)

            # Find all circles using the opencv HoughCircles function
            circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=50,
                                       maxRadius=100)
            print(circles)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                chosen_circle = None
                for i in circles[0, :]:
                    if chosen_circle is None:
                        chosen_circle = i
                    if previousCircle is not None:
                        if distance(chosen_circle[0], chosen_circle[1], previousCircle[0], previousCircle[1]) <= distance(i[0], i[1], previousCircle[0], previousCircle[1]):
                            chosen_circle = i

                cv2.circle(frame, (chosen_circle[0], chosen_circle[1]), 1, (0, 255, 0), 3)
                cv2.circle(frame, (chosen_circle[0], chosen_circle[1]), chosen_circle[2], (0, 0, 255), 3)
                previousCircle = chosen_circle

            cv2.imshow('Balls', frame)
            key = cv2.waitKey(1) & 0XFF
            if key == 113:
                break
        video.release()
        cv2.destroyAllWindows()


obd = ObjectDetection()
# ob.detect_image('static/images/people_2.jpg')
# bBox = obd.detect_track_video('static/upload/PENALTY KICK.mp4')
# print(bBox)

# obd.yolov4_ball_detection('static/upload/11.mp4')

obd.ball_detection()
