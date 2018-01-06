
"""
This is a demo for running face recognition on live video from your web camera "cv2.VideoCapture(0)",(0) indicates
web camera is chosen.

Note: If you want to choose a specific camera change the index to 1 or 2.

This demo has two parts:

1. Function - train_kNN
2. Face_Recogntion Script

Process:

1. Call the train_kNN() script.

Note: It is assumed that you have "People_Directory" in the same folder, you can name
it something else, considering you also change the train_dir name as well.

It is also important to note that you need to call the train_kNN function just once, after which you can comment it out.

2. Face_Recogntion Script :

 This script loads up the learn kNN-classifier model and computes distance metric on each frame. Once the closest
 distance results are obtained it displays the result as a name of the person along with a bounding box.


"""



import face_recognition
import cv2
import pickle
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder




ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}

def train_kNN(train_dir = "People_Directory", model_save_path = "kNN", n_neighbors = None, knn_algo = 'ball_tree',
              verbose=False):
    # """
    # Trains a k-nearest neighbors classifier for face recognition.
    #
    # :param train_dir: directory - "People_Directory" should contain sub-directories for each known person, with its
    #  name.

    #  Example Structure:
    #
    # ---- Person_A : Image1, Image2,......
    # ---- Person_B : Image1, Image2,......
    # ---- Person_c : Image1, Image2,......
    #         --
    #         --
    #
    # :param model_save_path: path to save model of disk; this file is created automatically.
    # :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not
    #  specified.
    # :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    # :param verbose: verbosity of training
    # :return: returns knn classifier that was trained on the given data.
    # """


    X = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes)
                                                                                   < 1 else "found more than one face"))
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)


    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    if model_save_path != "":
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf

# Call train_kNN model before running the script
#
#train_kNN()


# Initialize some variables
face_encodings = []
face_names = []
process_this_frame = True


# Setup the camera module

video_capture = cv2.VideoCapture(0)

# Load the learned model

with open("kNN", 'rb') as f:
    kNN_clf = pickle.load(f)

# Loop in each frame and compute classification for known faces


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing

    #print(X_img)
    cv2.imwrite("Frame.jpg", frame)
    X_img = face_recognition.load_image_file("Frame.jpg")
    X_faces_loc = face_locations(X_img)

    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)


    if faces_encodings!=[]:
        closest_distances = kNN_clf.kneighbors(faces_encodings, n_neighbors=1)

        DIST_THRESH = .5
        is_recognized = [closest_distances[0][i][0] <= DIST_THRESH for i in range(len(X_faces_loc))]



        for pred, loc, rec in zip(kNN_clf.predict(faces_encodings), X_faces_loc, is_recognized):

        # predict classes and cull classifications that are not with high confidence


            # (top, right, bottom, left) => (left,top,right,bottom)
            # Draw a box around the face
            #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (loc[3], loc[0]), (loc[1],loc[2]), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (loc[3], loc[2] - 35), (loc[1], loc[2]), (0, 0,255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, pred, (loc[3]+ 6, loc[2] - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the camera
video_capture.release()
cv2.destroyAllWindows()
