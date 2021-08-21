"""
-------------------------------------------------
This script is used to read the data in the tfrecord file and decode it.

-------------------------------------------------
"""
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
import cv2
import numpy as np



def pares_tf(example_proto):
    # Define the dictionary used to parse
    dics = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'filename': tf.FixedLenFeature([], tf.string),                           
        'encoded': tf.FixedLenFeature([], tf.string),
        'format': tf.FixedLenFeature([], tf.string),                               
        'class': tf.FixedLenSequenceFeature([], tf.string,allow_missing=True),
        'truncated': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'occluded': tf.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
        'alpha': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'xmin': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'xmax': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'ymin': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'ymax': tf.FixedLenSequenceFeature([], tf.float32,allow_missing=True)
    }
    # Calling the interface to parse an example
    parsed_example = tf.parse_single_example(example_proto,dics)
    image = tf.image.decode_png(parsed_example['encoded'], channels=3)
    filename = parsed_example['filename']
    label = parsed_example['class']
    width = parsed_example['width']
    height = parsed_example['height']
    xmin = parsed_example['xmin']
    xmax = parsed_example['xmax']
    ymin = parsed_example['ymin']
    ymax = parsed_example['ymax']
    image = tf.image.resize(image, (height,width))
    image = tf.cast(image,tf.float32)
    return image,label,width,height,xmin,xmax,ymin,ymax,filename

# Convert parsed results to array format
def tensor_to_array(parsed_record):
    array = []
    for it in range(len(parsed_record)):
        array.append(parsed_record[it].numpy())
    return array