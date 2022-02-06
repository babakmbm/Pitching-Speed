import sys
import cv2
import os
from ObjectTracking import ObjectTracking

class ObjectDetection:
    def __init__(self):
        if os.path.isfile('static/model/HAAR_CASCADES/fullbody.xml'):
            self.detector = cv2.CascadeClassifier('static/model/HAAR_CASCADES/fullbody.xml')
        else:
            print('Haar cascade could not loaded!')
            self.detector = False
            sys.exit()


    def detect_image(self, image_path):

        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.detector:
            detections = self.detector.detectMultiScale(image_gray)
            print(detections)
            print(len(detections), "people detected!\n")

            for (x, y, width, height) in detections:
                print(x, y, width, height)
                cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 1)
                cv2.imshow('Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def detect_track_video(self, video_path):
        tr = ObjectTracking("CSRT", video_path)
        tracker = tr.track_with('CSRT')
        #print(tracker)
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
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow('Detection-Tracking', frame)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                if x > 0:
                    print('Detection Successful!')
                    return x, y, w, h

'''OD = ObjectDetection()
#ob.detect_image('static/images/people_2.jpg')
bBox = OD.detect_track_video('static/upload/walking.avi')
print(bBox)'''