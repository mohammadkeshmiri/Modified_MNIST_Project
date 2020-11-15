import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.model_selection as sk

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def CSVToTFRecord(x_fileName, y_fileName, outputTrain_fileName, outputTest_fileName):
    """
    Create a tfrecord file.
    
    Args:
        image_data (List[(image_file_path (str), label (int), instance_id (str)]): the data to store in the tfrecord file. 
        The `image_file_path` should be the full path to the image, accessible by the machine that will be running the 
        TensorFlow network. The `label` should be an integer in the range [0, number_of_classes). `instance_id` should be 
        some unique identifier for this example (such as a database identifier). 
        output_path (str): the full path name for the tfrecord file. 
    """
    x_csv = pd.read_csv(x_fileName, header=None).values
    y_csv = pd.read_csv(y_fileName, header=None).values

    X_train, X_test, y_train, y_test = splitData(x_csv, y_csv)

    writeTFRecord(X_train, y_train, outputTrain_fileName)
    writeTFRecord(X_test, y_test, outputTest_fileName)

def splitData(features, labels):
    X_train, X_test, y_train, y_test = sk.train_test_split(features,labels,test_size=0.2, random_state = 42)
    return X_train, X_test, y_train, y_test
    
def writeTFRecord(features, labels, fileName):
    writer = tf.python_io.TFRecordWriter(fileName)

    numOfRows = np.size(labels, 0)
    for i in range(0, numOfRows):
        image = features[i]
        label = labels[i,0]
        example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': _float_feature([label]),
            'image': _float_feature(image)
        }
        ))

        writer.write(example.SerializeToString())

    writer.close()

CSVToTFRecord('train_x.csv', 'train_y.csv', 'my_train2.tfrecord', 'my_test2.tfrecord')