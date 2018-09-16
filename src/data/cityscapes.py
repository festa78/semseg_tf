from collections import namedtuple
import os

import numpy as np
import glob
import tensorflow as tf

import project_root

# yapf: disable
# The cityscapes label information folked from:
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

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

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
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

name2label      = { label.name    : label for label in labels           }
id2label        = { label.id      : label for label in labels           }
trainId2label   = { label.trainId : label for label in reversed(labels) }
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]
# yapf: enable


def get_file_path(input_dir):
    """Parse Cityscapes data and get file path list.

    Parameters
    ----------
    input_dir: str
        The directory path of Cityscapes data.
        Assume original file structure.

    Returns
    -------
    data_list: dict
        The dictinary which contains a list of image and label pair.
    """
    # Constants.
    IMAGE_ROOT = 'leftImg8bit'
    LABEL_ROOT = 'gtFine'
    LABEL_SURFIX = '_labelIds.png'
    DATA_CATEGORY = ['train', 'val', 'test']

    input_dir = os.path.abspath(os.path.expanduser(input_dir))

    data_list = {}
    for category in DATA_CATEGORY:
        image_list = glob.glob(
            os.path.join(input_dir, IMAGE_ROOT, category, '*/*'))

        # Get label path corresponds to each image path.
        label_list = []
        for f in image_list:
            root_path = os.path.join(input_dir, LABEL_ROOT, category)
            area_name = f.split('/')[-2]
            base_name = os.path.basename(f).replace(IMAGE_ROOT,
                                                    LABEL_ROOT).replace(
                                                        '.png', LABEL_SURFIX)
            label_list.append(os.path.join(root_path, area_name, base_name))

        data_list[category] = {
            'image_list': image_list,
            'label_list': label_list
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
    label_trainid =  tf.py_func(
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