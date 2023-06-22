import face_recognition
import os
import argparse
import json

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-o", "--output", required=True,
	help="path to directory of output facial encodings JSON file")
ap.add_argument("-m", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either 'hog' or 'cnn'")
args = vars(ap.parse_args())

# calculates facial encodings of people in {ds_dir} dataset
# dataset directory is expected to have subdirectories, each containing person photos
def encode(data, ds_dir, det_method):
    i = 0
    for person_dir in os.listdir(ds_dir):
        encodings = []
        for image_name in os.listdir(f'{ds_dir}/{person_dir}'):
            image = face_recognition.load_image_file(f"{ds_dir}/{person_dir}/{image_name}")
            boxes = face_recognition.face_locations(image, model=det_method)
            if not boxes:
                print(f"Warning: Failed to detect face in file {person_dir}/{image_name}")
                continue
            encodings.append(face_recognition.face_encodings(image, boxes)[0].tolist())
        data[str(i)] = {"name": person_dir.replace("_", " ").title(), "encoding": encodings}
        i += 1

if __name__ == '__main__':
    data = {}

    print("Start encoding...")
    encode(data, args["input"], args["detection_method"])

    with open(f"{args['output']}/encodings.json", "w") as write_file:
        json.dump(data, write_file)
