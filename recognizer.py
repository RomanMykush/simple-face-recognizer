import face_recognition
import cv2 as cv
import numpy as np
import threading

def RecognizeFace(data, frame, det_method="hog"):
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # detecting faces
    boxes = face_recognition.face_locations(rgb, model=det_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    names = ["Unknown" for i in range(len(boxes))]
    confidences = [-1 for i in range(len(boxes))]
    # recognizing faces
    for i in range(len(encodings)):
        for key, value in data.items():
            matches = face_recognition.compare_faces(np.asarray(value["encoding"]), encodings[i])
            distances = face_recognition.face_distance(np.asarray(value["encoding"]), encodings[i])

            best_match_index = np.argmin(distances)
            # storing best matches with smallest differences
            if matches[best_match_index] and distances[best_match_index] < (1 - confidences[i]):
                names[i] = value["name"]
                confidences[i] = distances[best_match_index]
        if confidences[i] != -1:
            names[i] += " (%.2f)" % (1 - round(confidences[i], 2))
    return boxes, names

def DrawFaceRectangle(frame, boxes, names):
    for (top, right, bottom, left), name in zip(boxes, names):
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 2)
        cv.rectangle(frame, (left, bottom + 32), (right, bottom), (0, 0, 200), cv.FILLED)
        cv.putText(frame, name, (left + 6, bottom + 20), cv.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
    return frame

class VideoFaceRecognizer:
    last_boxes = []
    last_names = []

    buffer_boxes = []
    buffer_names = []

    encode_thread = threading.Thread()
    det_method = "hog"

    def __init__(self, data):
        self.data = data

    def run_encode_thread(self, frame):
        self.buffer_boxes, self.buffer_names = RecognizeFace(self.data, frame, self.det_method)

    def run_recognition(self, input, det_method="hog"):
        self.det_method = det_method

        if input.isdigit():
            input = int(input)
        # create reader for video file or webcam stream
        cap = cv.VideoCapture(input)
        if type(input) != int:  # get frame rate of video file
            delay = int(1000 / cap.get(cv.CAP_PROP_FPS))
        else:   # 30 fps for webcam
            delay = int(1000 / 30)

        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            # check if encode_thread has finished its job
            if not self.encode_thread.is_alive():
                # storing thread results
                self.last_boxes = self.buffer_boxes
                self.last_names = self.buffer_names
                # creating and starting new thread
                self.encode_thread = threading.Thread(target=self.run_encode_thread, args=[frame])
                self.encode_thread.start()
            
            cv.imshow('Video', DrawFaceRectangle(frame, self.last_boxes, self.last_names))
            # check if exit key was pressed or window was closed
            if cv.waitKey(delay) == ord('q') or cv.getWindowProperty('Video', cv.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv.destroyAllWindows()

class ImageFaceRecognizer:
    def __init__(self, data):
        self.data = data

    def run_recognition(self, input, det_method="hog"):
        # reading image from file
        image = cv.imread(input)
        # getting faces rectangle and corresponding names
        boxes, names = RecognizeFace(self.data, image, det_method)
        cv.imshow('Image', DrawFaceRectangle(image, boxes, names))
        # check if exit key was pressed or window was closed
        while True:
            if cv.waitKey(100) == ord('q') or cv.getWindowProperty('Image', cv.WND_PROP_VISIBLE) < 1:
                cv.destroyAllWindows()
                break