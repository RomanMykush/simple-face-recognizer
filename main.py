from recognizer import VideoFaceRecognizer, ImageFaceRecognizer
import argparse
import json

# supported file formats
SUPP_VIDEO_FRT = ["avi", "mp4", "mkv", "wmv"]
SUPP_IMG_FRT = ["png", "webp", "jpg", "jpeg", "jpe", "bmp", "dib", "jp2", "pbm", "pgm", "ppm", "pxm", "pnm" "sr", "ras", "tiff", "tif", "exr", "hdr", "pic"]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input test photo, video or webcam number")
ap.add_argument("-e", "--encodings", required=True,
	help="path to file of facial encodings")
ap.add_argument("-m", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

def create_recognizer(data, input):
    if input.isdigit():             # webcam
        return VideoFaceRecognizer(data)
    format = input.split(".")[-1]
    if format in SUPP_VIDEO_FRT:    # video
        return VideoFaceRecognizer(data)
    if format in SUPP_IMG_FRT:      # image
        return ImageFaceRecognizer(data)
    raise ValueError('Invalid file format passed as argument')

if __name__ == '__main__':
    # get known face encodings data
    with open(args["encodings"], "r") as read_file:
        data = json.load(read_file)
    # start face recognition in file (or stream)
    fr = create_recognizer(data, args["input"])
    print("Running face recognizer...")
    fr.run_recognition(args["input"])
