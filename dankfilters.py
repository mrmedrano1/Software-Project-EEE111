import numpy as np
import cv2

#converts overlay to alpha channel and blends foreground to background
def convert_overlay(source, overlay, position=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = source.shape
    y, x = position[0], position[1] #position of the foreground/background
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alphachannel = float(overlay[i][j][3] / 255.0)
            source[x + i][y + j] = alphachannel * overlay[i][j][:3] + (1 - alphachannel) * source[x + i][y + j]
    return source

#functions for resizing the filters
def minsize(y, minhfactor, h, n):
    smin = int(y + minhfactor * h / n)
    return smin

def maxsize(y, maxhfactor, h, n):
    smax = int(y + maxhfactor * h / n)
    return smax

def fsize(maximum_size, minimum_size):
    fsize = maximum_size - minimum_size
    return fsize

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_filter = cv2.imread('shades.png', -1)
mouth_filter = cv2.imread('moustache.png',-1)

#start of livefeed

capture = cv2.VideoCapture(0)

while True:
    ret, livefeed = capture.read()
    gray = cv2.cvtColor(livefeed, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(livefeed, 1.2, 5, 0, (120,120), (300, 300))
    for (x, y, w, h) in face:
        if h > 0 and w > 0:

            eye_min = minsize(y, 1.2, h, 5)
            eye_max = maxsize(y, 2.9, h, 5)
            eye_sz = fsize(eye_max, eye_min)
 
            mouth_min = minsize(y, 3.3, h, 6)
            mouth_max = maxsize(y, 6, h, 6)
            mouth_sz = fsize(mouth_max, mouth_min)

            eye_overlayed = livefeed[eye_min:eye_max, x:x+w]
            mouth_overlayed = livefeed[mouth_min:mouth_max, x:x+w]

            eye = cv2.resize(eye_filter, (w, eye_sz),interpolation=cv2.INTER_CUBIC)
            mouth = cv2.resize(mouth_filter, (w, mouth_sz),interpolation=cv2.INTER_CUBIC)

            convert_overlay(eye_overlayed, eye)
            convert_overlay(mouth_overlayed, mouth)

    cv2.imshow('Dank Filters', livefeed)
    if cv2.waitKey(1) == ord('q'):
        break
 
capture.release()

cv2.destroyAllWindows()
