import tensorflow as tf
import numpy as np

def input_fn(filenames, isTrain, batch_size=32, buffer_size=2048, epoch = 1):
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse_function)

    if isTrain:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = None

    else:
        # If testing then don't shuffle the data.
        
        # Only go through the data once.
        num_repeat = None

        
    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)
    
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = images_batch
    y = labels_batch

    return x, y

# example proto decode
def parse_function(example_proto):
    keys_to_features = {'image':tf.FixedLenFeature(([1,4096]), tf.float32),
                        'label': tf.FixedLenFeature((), tf.float32, default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    return parsed_features['image'], parsed_features['label']

def one_hot_labels(labels):

    possibleLabels = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81])
    possibleLabels = tf.reshape(possibleLabels, [40,1])

    condition = tf.equal(possibleLabels, labels)
    condition = tf.transpose(condition, [1,0])
    newLabel = tf.where(condition)
    newLabel = newLabel[:,1]
    return tf.one_hot(newLabel, depth = 40)