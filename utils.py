import cv2
import os
import sys
from glob import glob


def video_to_frames(video_path):
    save_path = 'static/frames'
    files = glob(save_path + '/*.jpg')
    pre_files_num = len(files)

    video = cv2.VideoCapture(video_path)
    i = pre_files_num+1
    if not video.isOpened():
        print('Could not load video!')
        sys.exit()

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        else:
            cv2.imwrite(save_path + '/frame_' + str(i) + '.jpg', frame)
            i += 1
    print(f"{i - pre_files_num} frames are created and saved to {save_path}")
    video.release()
    cv2.destroyAllWindows()
