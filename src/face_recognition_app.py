from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
import numpy as np
import facenet
import imutils
import os
import sys
import math
import pickle
import align.detect_face
import cv2
import collections
from sklearn.svm import SVC
import streamlit as st

def euclidean_distance(embedding1, embedding2):
    return np.sqrt(np.sum(np.square(embedding1 - embedding2)))

def load_model():
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            facenet.load_model(FACENET_MODEL_PATH)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            return sess, images_placeholder, embeddings, phase_train_placeholder, embedding_size

def main():
    st.title("Real-Time Face Recognition App")

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'

    # Load the custom classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    st.write("Custom Classifier, Successfully loaded")

    sess, images_placeholder, embeddings, phase_train_placeholder, embedding_size = load_model()
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    # Initialize known embeddings dictionary
    known_embeddings = {}
    for i, class_name in enumerate(class_names):
        class_images = []  # Add class images here
        embeddings_list = []
        for img in class_images:
            scaled = cv2.resize(img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            scaled = facenet.prewhiten(scaled)
            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
            embeddings_list.append(emb_array[0])
        known_embeddings[class_name] = np.mean(embeddings_list, axis=0)

    person_detected = collections.Counter()
    cap = VideoStream(src=0).start()

    frame_window = st.image([])

    while True:
        frame = cap.read()
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)

        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]

        if faces_found > 0:
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                if (bb[i][3] - bb[i][1]) / frame.shape[0] > 0.25:
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    if cropped.size > 0:
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)

                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]

                        if best_class_probabilities[0] > 0.7:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            name = class_names[best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                        else:
                            min_dist = float('inf')
                            closest_name = "Unknown"
                            for known_name, known_embedding in known_embeddings.items():
                                dist = euclidean_distance(emb_array[0], known_embedding)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_name = known_name

                            if min_dist < 1.0:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.putText(frame, closest_name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                                cv2.putText(frame, str(round(min_dist, 3)), (text_x, text_y + 17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                            else:
                                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20
                                cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=1, lineType=2)

        frame_window.image(frame)

        if st.button("Stop"):
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
