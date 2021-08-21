"""
-------------------------------------------------
This script is used to perform object detection using the trained
model, outputting the detection results in image format and run time.

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
import datetime
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './models/416/',
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
   
    print("Loading the model begins: ",datetime.datetime.now())
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    print("End of loading model: ",datetime.datetime.now())
    
    # image_list = np.array([7,8])
    fps = 0
    mean_ts = 0.
    for image_num in range(51):
        load_time1 = datetime.datetime.now()
        img_num = str(image_num).zfill(6)
        # output path and filename
        image_path = os.path.join("./training/image_2/",img_num + '.png')
        output = os.path.join("./results/","result_test_" + img_num + '.png')
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        image_data = utils.image_preprocess(np.copy(original_image),[input_size, input_size])
        
        
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        load_time2 = datetime.datetime.now()
        print("time of loading images: ",load_time2-load_time1)
        cal_time1 = time.time()
        batch_data = tf.constant(images_data)
        # batch_data1 = tf.constant(images_data1)
       
        pred_bbox = infer(batch_data)
        cal_time2 = time.time()
        # pred_bbox1 = infer(batch_data1)
        # print("计算结束时间1：",datetime.datetime.now())
        # pred_bbox2 = infer(batch_data)
        # print("计算结束时间2：",datetime.datetime.now())
        # print(pred_bbox)
        cal_time = cal_time2-cal_time1
        print("Calculating time: ",cal_time)
        # cal_ts = float(time.mktime(time.strptime(cal_time, "%Y-%m-%d %H:%M:%S")))
        
        pre_time1 = time.time()
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
        pre_time2 = time.time()
        pre_time = pre_time2-pre_time1
        print("Post-processing time: ",pre_time)
        # pre_ts = float(time.mktime(time.strptime(pre_time, "%Y-%m-%d %H:%M:%S")))
        
        draw_time1 = datetime.datetime.now()
        image = utils.draw_bbox(original_image, pred_bbox,FLAGS.size)
        # image = utils.draw_bbox(image_data*255, pred_bbox)
        image = Image.fromarray(image.astype(np.uint8))
        #image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output, image)
        draw_time2 = datetime.datetime.now()
        print("time of drawing bboxes and saving pictures: ",draw_time2-draw_time1)

        if(image_num > 0):
            mean_ts = mean_ts + cal_time + pre_time
    
    
    mean_ts = mean_ts / 50.
    fps = 1/mean_ts
    print("average time: ",mean_ts)
    print("fps：", fps)
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
