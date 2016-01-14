import cv2
import sys
import threading
from tkinter import *

cap = cv2.VideoCapture(0)
minval = 0
maxval = 10


def image_processing():
    hand_classifier = cv2.CascadeClassifier("haarcascade/palm.xml")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        hands = hand_classifier.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
        )

        edges = cv2.Canny(gray, minval, maxval)

        # Display the resulting frame
        cv2.imshow('out', edges)
        cv2.imshow('in', frame)

        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('hands', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def set_min(event):
    global minval
    minval = int(event)


def set_max(event):
    global maxval
    maxval = int(event)


image_processing_thread = threading.Thread(None, image_processing)
image_processing_thread.start()
root = Tk()

Label(root, text="Min").pack()
slider_min = Scale(root,  # родительское окно
                   from_=0,
                   to=100,
                   command=set_min,
                   bg="white", fg="black")  # цвет фона и надписи
slider_min.pack()  # расположить кнопку на главном окне

Label(root, text="Max").pack(padx=5, pady=10, side=LEFT)
slider_max = Scale(root,  # родительское окно
                   from_=0,
                   to=100,
                   command=set_max,
                   bg="white", fg="black")  # цвет фона и надписи
slider_max.pack(padx=5, pady=10, side=LEFT)

root.mainloop()
image_processing_thread.join()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
