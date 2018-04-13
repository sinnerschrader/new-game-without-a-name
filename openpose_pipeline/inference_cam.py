'''
All code is highly based on Ildoo Kim's code (https://github.com/ildoonet/tf-openpose)
and derived from the OpenPose Library (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''

import tensorflow as tf
import cv2
from tensorflow.core.framework import graph_pb2
import time

from common import estimate_pose, draw_humans, preprocess

if __name__ == '__main__':
    t0 = time.time()

    cam = cv2.VideoCapture(0)

    t1 = time.time()
    print("Time {} for cam record".format(t1))

    tf.reset_default_graph()

    graph_def = graph_pb2.GraphDef()

    with open('models/optimized_openpose.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')

    t2 = time.time()
    print("Time {} for import graph def".format(t2 - t1))

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

    t3 = time.time()
    print("Time {} for tensor creation".format(t3 - t2))

    with tf.Session() as sess:
        while cam.isOpened():
            success, image = cam.read()
            t4 = time.time()
            print("Time {} for reading cam".format(t4 - t3))

            if not success:
                continue

            image = preprocess(image, 656, 368)

            t5 = time.time()
            print("Time {} for preprocessed image".format(t5 - t4))

            # The session runner is really slow
            heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
                inputs: image
            })

            t6 = time.time()
            print("Time {} for session runner".format(t6 - t5))

            heatMat, pafMat = heatMat[0], pafMat[0]

            humans = estimate_pose(heatMat, pafMat)

            # display
            success, image = cam.read()
            image_h, image_w = image.shape[:2]
            image = draw_humans(image, humans)

            scale = 480.0 / image_h
            newh, neww = 480, int(scale * image_w + 0.5)

            image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

            cv2.imshow('result', image)

            t7 = time.time()
            print("Time {} for scaling and showing the image".format(t6 - t5))

            # Press Q to exit the video playback
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
