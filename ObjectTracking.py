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
        return tracker

    def track_video(self):
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
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Tracking Failed!', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                cv2.putText(frame, f'ALGORITHM: {self.tracking_algorithm}', (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow('Tracking Window', frame)
                if cv2.waitKey(1) & 0XFF == 27:
                    break
        else:
            print('Error while loading the frame!')
            sys.exit()

# test section
# ---'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT'----
# tr = ObjectTracking("CSRT", 'static/upload/Ankara neymar.mp4')
# tr.track_video()
