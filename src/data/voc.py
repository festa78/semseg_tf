from collections import namedtuple
import os

import numpy as np
import glob
import tensorflow as tf

import project_root

# yapf: disable
# The voc label information following the format from:
# https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # An integer ID that overwrites the ID above, when creating ground truth
                    # images for training.
                    # For training, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 0 during training.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   color
    Label(  'background'           ,  0 ,        0 , (  0,  0,  0) ),
    Label(  'aeroplane'            ,  1 ,        1 , (128,  0,  0) ),
    Label(  'bicycle'              ,  2 ,        2 , (  0,128,  0) ),
    Label(  'bird'                 ,  3 ,        3 , (128,128,  0) ),
    Label(  'boat'                 ,  4 ,        4 , (  0,  0,128) ),
    Label(  'bottle'               ,  5 ,        5 , (128,  0,128) ),
    Label(  'bus'                  ,  6 ,        6 , (  0,128,128) ),
    Label(  'car'                  ,  7 ,        7 , (128,128,128) ),
    Label(  'cat'                  ,  8 ,        8 , ( 64,  0,  0) ),
    Label(  'chair'                ,  9 ,        9 , (192,  0,  0) ),
    Label(  'cow'                  , 10 ,       10 , ( 64,128,  0) ),
    Label(  'diningtable'          , 11 ,       11 , (192,128,  0) ),
    Label(  'dog'                  , 12 ,       12 , ( 64,  0,128) ),
    Label(  'horse'                , 13 ,       13 , (192,  0,128) ),
    Label(  'motorbike'            , 14 ,       14 , ( 64,128,128) ),
    Label(  'person'               , 15 ,       15 , (192,128,128) ),
    Label(  'pottedplant'          , 16 ,       16 , (  0, 64,  0) ),
    Label(  'sheep'                , 17 ,       17 , (128, 64,  0) ),
    Label(  'sofa'                 , 18 ,       18 , (  0,192,  0) ),
    Label(  'train'                , 19 ,       19 , (128,192,  0) ),
    Label(  'tvmonitor'            , 20 ,       20 , (  0, 64,128) ),
    Label(  'undefined'            ,255 ,      255 , (  0,  0,  0) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

name2label      = { label.name    : label for label in labels           }
id2label        = { label.id      : label for label in labels           }
trainId2label   = { label.trainId : label for label in reversed(labels) }
# yapf: enable


def get_file_path(input_dir, train_list, val_list):
    """Parse data and get file path list.

    Parameters
    ----------
    input_dir: str
        The directory path of Cityscapes data.
        Assume original file structure.
    train_list: str
        The list of train file basenames.
    val_list: str
        The list of val file basenames.

    Returns
    -------
    data_list: dict
        The dictinary which contains a list of image and label pair.
    """
    # Constants.
    IMAGE_ROOT = 'JPEGImages'
    LABEL_ROOT = 'SegmentationClassAug'

    image_dir = os.path.join(
        os.path.abspath(os.path.expanduser(input_dir)), IMAGE_ROOT)
    label_dir = os.path.join(
        os.path.abspath(os.path.expanduser(input_dir)), LABEL_ROOT)

    assert os.path.exists(
        label_dir
    ), 'Need to download SegmentationClassAug from https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip'

    train_image_list = [os.path.join(image_dir, f + '.jpg') for f in train_list]
    train_label_list = [os.path.join(label_dir, f + '.png') for f in train_list]
    val_image_list = [os.path.join(image_dir, f + '.jpg') for f in val_list]
    val_label_list = [os.path.join(label_dir, f + '.png') for f in val_list]

    data_list = {}
    data_list['train'] = {
        'image_list': train_image_list,
        'label_list': train_label_list
    }
    data_list['val'] = {
        'image_list': val_image_list,
        'label_list': val_label_list
    }
    return data_list


def id2trainid_tensor(label):
    """Make tf.PyFunc which converts id to trainId.

    Parameters
    ----------
    label: (H, W, C=1) or (N, H, W, C=1) tensor
        Each element has id.

    Returns
    -------
    tf.PyFunc
        functional which converts id to trainId.

    """

    label_trainid = tf.py_func(
        func=lambda x: np.array([id2label[int(i)].trainId for i in np.nditer(x)], dtype=np.int64).reshape(x.shape),
        inp=[label],
        Tout=tf.int64)

    # Restore shape as py_func loses shape information.
    label_trainid.set_shape(label.get_shape())
    return label_trainid


def trainid2color_tensor(label):
    """Make tf.PyFunc which converts id to color.

    Parameters
    ----------
    label: (H, W, C=1) or (N, H, W, C=1) tensor
        Each element has id.
        The last dimension must be a channel axis.

    Returns
    -------
    tf.PyFunc
        functional which converts id to color.

    """

    def func(x):
        x_shape = x.shape
        x_shape = [item for item in x_shape[:-1]] + [3]
        return np.array(
            [trainId2label[int(i)].color for i in np.nditer(x)],
            dtype=np.float32).reshape(x_shape)

    label_color = tf.py_func(func=func, inp=[label], Tout=tf.float32)

    # Restore shape as py_func loses shape information.
    shape = label.get_shape()
    shape = [item for item in shape[:-1]] + [3]
    label_color.set_shape(shape)

    return label_color
