"""
-------------------------------------------------
This script is used to store the data relevant to the kitti dataset in 
TFRecord format. Also in this script the dataset is divided into training
set and validation set and stored separately.
   
-------------------------------------------------
"""

import io
import os
import numpy as np
import PIL.Image as pil
from PIL import Image
import tensorflow as tf
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./training/',
                    help='Location of the kitti dataset')
parser.add_argument('--output_path', type=str, default='./training/kitti_tfrecords/',
                    help='Output location of the TFRecord file')
parser.add_argument('--classes_to_use', default='car,van,truck,pedestrian,person_sitting,cyclist,tram', 
                    help='Classes to be detected in KITTI')

def convert_kitti_to_tfrecords(data_dir, output_path,classes_to_use):
    """
    Convert KITTI detections to TFRecords.
    :param data_dir: Source data directory
    :param output_path: Output file directory
    :param classes_to_use: Classes to be detected in KITTI
    :return:
    """
    train_count = 0
    val_count = 0

    annotation_dir = os.path.join(data_dir,
                                  'label_2')
    image_dir = os.path.join(data_dir,
                             'image_2')
    # The train. txt stores the image names of the training set
    it = open(data_dir+'train.txt')
    train_text = it.read().splitlines()
    train_writer = tf.io.TFRecordWriter(output_path + 'train.tfrecord')
    val_writer = tf.io.TFRecordWriter(output_path + 'val.tfrecord')

    # 
    images = sorted(os.listdir(image_dir))
    for img in images:

      
        img_num = str(int(img.split('.')[0])).zfill(6)
        img_anno = read_annotation_file(os.path.join(annotation_dir,
                                                     img_num + '.txt'))

        # Filter classes
        # Filter out some useless classes and annotations in the dontcare area
        annotation_for_image = filter_annotations(img_anno, classes_to_use)
        
        image_path = os.path.join(image_dir, img)
        example = prepare_example(image_path, annotation_for_image)
        # Determining whether to write to the training set or the validation set
        if img_num in train_text:
            train_writer.write(example.SerializeToString())
            train_count += 1
        else:
            val_writer.write(example.SerializeToString())
            val_count += 1

    train_writer.close()
    val_writer.close()

def filter_annotations(img_all_annotations, used_classes):
    """
    Filter out some useless classes and annotations in the dontcare area
    :param img_all_annotations: All annotations of the picture
    :param used_classes: Need to keep
    :return:
    """
    img_filtered_annotations = {}  
    relevant_annotation_indices = [
        i for i, x in enumerate(img_all_annotations['type']) if x in used_classes
    ]
    # print(relevant_annotation_indices)
    for key in img_all_annotations.keys():
        img_filtered_annotations[key] = (
            img_all_annotations[key][relevant_annotation_indices])


    return img_filtered_annotations


def read_annotation_file(filename):
    """
    Function reading annotation files
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]
    anno = dict()
    anno['type'] = np.array([x[0].lower() for x in content])
    anno['truncated'] = np.array([float(x[1]) for x in content])
    anno['occluded'] = np.array([int(x[2]) for x in content])
    anno['alpha'] = np.array([float(x[3]) for x in content])

    anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
    anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
    anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
    anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    return anno

# The following functions are used to define the format of some 
# protocol examples for using in construction
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
 
def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
 
 
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
 
def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
 
 
def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def prepare_example(image_path, annotations):
    """
    Annotations of an image into tf.Example proto.
    :param image_path:
    :param annotations:
    :return:
    """
    # Read the content of an image and convert it to an array format
    with open(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = pil.open(encoded_png_io)
    image = np.asarray(image)
    
    # Coordinate processing
    width = int(image.shape[1])
    height = int(image.shape[0])

    xmin_norm = annotations['2d_bbox_left']
    ymin_norm = annotations['2d_bbox_top']
    xmax_norm = annotations['2d_bbox_right']
    ymax_norm = annotations['2d_bbox_bottom']
    
    
    classes_text = [x.encode('utf8') for x in annotations['type']]

    # Construct the protocol example
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(height),
        'width': int64_feature(width),
        'filename': bytes_feature(image_path.encode('utf8')),
        'encoded': bytes_feature(encoded_png),
        'format': bytes_feature('png'.encode('utf8')),
        'class': bytes_list_feature(classes_text),
        'truncated': float_list_feature(annotations['truncated']),
        'occluded': int64_list_feature(annotations['occluded']),
        'alpha': float_list_feature(annotations['alpha']),
        'xmin': float_list_feature(xmin_norm),
        'xmax': float_list_feature(xmax_norm),
        'ymin': float_list_feature(ymin_norm),
        'ymax': float_list_feature(ymax_norm)
    }))

    return example

def main(args):

    convert_kitti_to_tfrecords(
        data_dir=args.data_dir,
        output_path=args.output_path,
        classes_to_use=args.classes_to_use.split(',')
        )


if __name__ == '__main__':

    args = parser.parse_args(sys.argv[1:])
       
    main(args)
