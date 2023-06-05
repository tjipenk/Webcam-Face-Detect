"""
This is an example of using the k-nearest-neighbors (KNN) algorithm for face recognition.

When should I use this example?
This example is useful when you wish to recognize a large set of known people,
and make a prediction for an unknown person in a feasible computation time.

Algorithm Description:
The knn classifier is first trained on a set of labeled (known) faces and can then predict the person
in a live stream by finding the k most similar faces (images with closet face-features under euclidean distance)
in its training set, and performing a majority vote (possibly weighted) on their label.

For example, if k=3, and the three closest face images to the given image in the training set are one image of Biden
and two images of Obama, The result would be 'Obama'.

* This implementation uses a weighted vote, such that the votes of closer-neighbors are weighted more heavily.

Usage:

1. Prepare a set of images of the known people you want to recognize. Organize the images in a single directory
   with a sub-directory for each known person.

2. Then, call the 'train' function with the appropriate parameters. Make sure to pass in the 'model_save_path' if you
   want to save the model to disk so you can re-use the model without having to re-train it.

3. Call 'predict' and pass in your trained model to recognize the people in a live video stream.

NOTE: This example requires scikit-learn, opencv and numpy to be installed! You can install it with pip:

$ pip3 install scikit-learn
$ pip3 install numpy
$ pip3 install opencv-contrib-python

"""

import cv2,pafy
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw,ImageFont
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import time


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    
    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    font  = ImageFont.truetype("arial.ttf", 39)
    font2  = ImageFont.truetype("arial.ttf", 30)
    w, h = 420, 1090
    shape = [(0, 0), (w - 10, h - 10)]
    draw.rectangle(shape, fill ="#00090959")
    draw.text((10, 25), "Detected Name :", font=font)
    wordline=0
    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        text_height=25
        wordline=wordline+65
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name2 = name.encode("UTF-8")
        #font = ImageFont.truetype("arial.ttf", 50)
        font2=ImageFont.load_default()
        # Draw a label with a name below the face
        #text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 124, 255), outline=(0, 124, 255))
        draw.text((left + 6, bottom - text_height - 5), name,font=font2, fill=(255, 255, 255, 255))
        draw.text((10, wordline), name, font=font)

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


if __name__ == "__main__":
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")
    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Setting cameras up...')
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'

    #rtsp://<Username>:<Password>@<IP Address>:<Port>/cam/realmonitor?channel=1&subtype=0
    #channel: Channel, 1-8; subtype: Code-Stream Type, Main Stream 0, Extra Stream 1.
    #url = 'https://www.youtube.com/watch?v=lgkn1I0YRw8'
    url = 'rtsp://cctv:Test123!@10.66.1.1:554/cam/realmonitor?channel=2&subtype=0'
    #url='rtsp://184.72.239.149/vod/mp4:BigBuckBunny_175k.mov'
    #video = pafy.new(url)
    #videoplay = video.getbest(preftype="any")
    #print(cv2.getBuildInformation())
    #cap = cv2.VideoCapture(videoplay.url)
    cap = cv2.VideoCapture(url)
    prev_frame_time = 0
  
    # used to record the time at which we processed current frame
    new_frame_time = 0
    #cap = cv2.VideoCapture(0)
    while 1 > 0:
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Different resizing options can be chosen based on desired program runtime.
                # Image resizing for more stable streaming
                img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                process_this_frame = process_this_frame + 1
                if process_this_frame % 30 == 0:
                    predictions = predict(img, model_path="trained_knn_model.clf",distance_threshold=0.4)
                frame = show_prediction_labels_on_image(frame, predictions)
                #cv2.WINDOW_NORMAL makes the output window resizealbe
                cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
                #resize the window according to the screen resolution
                cv2.resizeWindow('camera', 600, 500)
                new_frame_time = time.time()
  
                # Calculating the fps
            
                # fps will be number of frame processed in given time frame
                # since their will be most of time error of 0.001 second
                # we will be subtracting it to get more accurate result
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
            
                # converting the fps into integer
                fps = int(fps)
            
                # converting the fps to string so that we can display it on frame
                # by using putText function
                fps = str(fps)
                gray=frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                # putting the FPS count on the frame
                cv2.putText(gray, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
                cv2.imshow('camera', frame)
        else:
            print("notcap")
        if ord('q') == cv2.waitKey(10):
            cap.release()
            cv2.destroyAllWindows()
            exit(0)
