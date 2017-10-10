import cv2
import dlib
import numpy as np


def get_face(frame, all_face=False):
    detector = dlib.get_frontal_face_detector()
    detects = detector(frame)





