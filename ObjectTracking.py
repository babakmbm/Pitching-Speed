import os.path
import sys
import cv2
from random import randint
import numpy as np


class ObjectTracking():
    def __init__(self, tracking_algorithm, video_path):
        self.tracking_algorithm = tracking_algorithm
        self.video_path = video_path

    def track_with(self, tracking_type):
        if tracking_type == 'BOOSTING' or 'Boosting' or 'boosting':
            tracker = cv2.legacy.TrackerBoosting_create()
        elif tracking_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        elif tracking_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        elif tracking_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        elif tracking_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        elif tracking_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        elif tracking_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()
        elif tracking_type == 'GOTURN':
            if not (os.path.isfile('goturn.caffemodel') and os.path.isfile('goturn.prototxt')):
                print("Error Loading GOTURN Model!")
                sys.exit()
            tracker = cv2.TrackerGOTURN_create()
        else:
            print("Invalid Tracker!")
        return tracker

    def track_video_single(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print('Cannot open file or video stream!')
            sys.exit()
        ret, frame = video.read()
        if ret:
            bBox = cv2.selectROI(frame)
            print(bBox)

            # choosing and initializing the algorithm
            tracker = self.track_with(self.tracking_algorithm)
            ret = tracker.init(frame, bBox)
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                ret, bBox = tracker.update(frame)

                if ret:
                    print(ret, bBox)
                    (x, y, w, h) = [int(v) for v in bBox]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Tracking Failed!', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                cv2.putText(frame, f'ALGORITHM: {self.tracking_algorithm}', (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 255), 2)
                cv2.imshow('Tracking Window', frame)
                if cv2.waitKey(1) & 0XFF == 27:
                    break
        else:
            print('Error while loading the frame!')
            sys.exit()

    def track_video_multi(self):
        video = cv2.VideoCapture(self.video_path)
        if not video.isOpened():
            print('Cannot open file or video stream!')
            sys.exit()
        ret, frame = video.read()

        bBoxes = []
        colors = []
        while True:
            bBox = cv2.selectROI('MultiTracker', frame)
            bBoxes.append(bBox)
            colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
            print('Press Q to quite and start the tracking')
            print('Press any key to select the next object!')
            key = cv2.waitKey(0) & 0XFF
            if key == 113:  # Q
                break

        print(bBoxes)
        print(colors)

        multi_tracker = cv2.legacy.MultiTracker_create()
        for bbox in bBoxes:
            multi_tracker.add(self.track_with(self.tracking_algorithm), frame, bbox)

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            ret, boxes = multi_tracker.update(frame)

            for i, new_box in enumerate(boxes):
                (x, y, w, h) = [int(v) for v in new_box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2)
                cv2.imshow('Multi tracker', frame)

            if cv2.waitKey(1) & 0XFF == 27:
                break


# test section
# ----'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT', 'GOTURN'----
tr = ObjectTracking("GOTURN", 'static/upload/PENALTY KICK.mp4')
tr.track_video_single()
