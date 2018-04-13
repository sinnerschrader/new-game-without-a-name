'''
All code is highly based on Ildoo Kim's code (https://github.com/ildoonet/tf-openpose)
and derived from the OpenPose Library (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''

import tensorflow as tf
import cv2
import argparse

from common import estimate_pose, draw_humans, read_imgfile

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/wywh.jpg')
    parser.add_argument('--input-width', type=int, default=656)
    parser.add_argument('--input-height', type=int, default=368)
    args = parser.parse_args()

    t0 = time.time()

    tf.reset_default_graph()
    
    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    # Download model from https://www.dropbox.com/s/2dw1oz9l9hi9avg/optimized_openpose.pb
    with open('models/optimized_openpose.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    t1 = time.time()
    print(t1 - t0)

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')

    t2 = time.time()
    print(t2 - t1)

    image = read_imgfile(args.imgpath, args.input_width, args.input_height)

    t3 = time.time()
    print(t3 - t2)

    with tf.Session() as sess:
        heatMat, pafMat = sess.run([heatmaps_tensor, pafs_tensor], feed_dict={
            inputs: image
        })

        t4 = time.time()
        print(t4 - t3)

        heatMat, pafMat = heatMat[0], pafMat[0]

        humans = estimate_pose(heatMat, pafMat)

        # display
        image = cv2.imread(args.imgpath)
        image_h, image_w = image.shape[:2]
        image = draw_humans(image, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

        cv2.imshow('result', image)
        t5 = time.time()
        print(t5 - t4)
        cv2.waitKey(0)
