#!/usr/bin/python
"""
-------------------------------------------------
This script is used to write the detection results into
txt format files to be used for evaluation.

-------------------------------------------------
"""
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

import os
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './models/416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_float('iou', 0.3, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    # using validation set to evaluate
    image_list = open("./training/val.txt", 'r')
    image_num_list = image_list.readlines()
    print(image_num_list)
    output_path = "./detect_results/416/"
    
    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    
    for img_num in image_num_list:
        img_num = img_num.replace('\n', '')
        image_path = os.path.join("./training/image_2/", img_num + '.png')
        # img_num = str(image_num).zfill(6)
        file_name = img_num + '.txt'
        file = open(output_path + file_name, 'w')
        # output = os.path.join("./result/","result_test_" + img_num + '.png')
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preprocess(np.copy(original_image),[input_size, input_size])
        # image_data = image_data / 100.
        # image_data = image_data[np.newaxis, ...].astype(np.float32)
        # 图片resize并normalization
        
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)

        # print(pred_bbox)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
            # print(pred_conf.shape)
            # print(boxes.shape)
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        xmin, ymin, xmax, ymax, score_list, classes_names = utils.write_results(original_image, pred_bbox,FLAGS.size)

        # for each detected bbox
        for i in range(len(score_list)):
            file.write(classes_names[i]) # which class type
            file.write(' ')
            file.write('-1') # truncation
            file.write(' ')
            file.write('-1')  # occlusion
            file.write(' ')
            file.write('-10')  # alpha
            file.write(' ')
            file.write(str(xmin[i]))  # bbox coor xmin
            file.write(' ')
            file.write(str(ymin[i]))  # bbox coor ymin
            file.write(' ')
            file.write(str(xmax[i]))  # bbox coor xmax
            file.write(' ')
            file.write(str(ymax[i]))  # bbox coor ymax
            file.write(' ')
            # 3D object dimensions: height, width, length
            file.write('-1 -1 -1 ')
            # 3D object location x, y, z
            file.write('-1000 -1000 -1000 ')
            file.write('-10 ')  # rotation
            file.write(str(score_list[i]))  # score
            file.write('\n')

        file.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
