"""Test set for voc classes.
"""

from PIL import Image
import numpy as np
import tensorflow as tf

from src.data.voc import get_file_path, id2trainid_tensor, trainid2color_tensor

TRAIN_FILENAMES = ['train1', 'train2', 'train3']
VAL_FILENAMES = ['val', 'val2']


def _create_sample_voc_structure(tmpdir):
    """Creates dummy voc like data structure.

    Returns
    -------
    root_dir_path : str
        Root path to the created data structure.
    data_list : dict
        Dummy data dictionary contains 'image_list' and 'label_list'.
    """
    # Constants.
    IMAGE_ROOT = 'JPEGImages'
    LABEL_ROOT = 'SegmentationClassAug'
    IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100

    np.random.seed(1234)

    data_list = {}
    tmpdir.mkdir(IMAGE_ROOT)
    tmpdir.mkdir(LABEL_ROOT)

    train_image_list = []
    train_label_list = []
    for filename in TRAIN_FILENAMES:
        # Creates empty files.
        image_path = tmpdir.join(IMAGE_ROOT, filename + '.jpg')
        label_path = tmpdir.join(LABEL_ROOT, filename + '.png')

        image = np.random.randint(255, size=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        image = Image.fromarray(image.astype(np.uint8))
        label = np.random.randint(10, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        label = Image.fromarray(label.astype(np.uint8))

        # Convert path from py.path.local to str.
        image.save(str(image_path))
        label.save(str(label_path))

        train_image_list.append(image_path)
        train_label_list.append(label_path)

    data_list['train'] = {
        'image_list': train_image_list,
        'label_list': train_label_list
    }

    val_image_list = []
    val_label_list = []
    for filename in VAL_FILENAMES:
        # Creates empty files.
        image_path = tmpdir.join(IMAGE_ROOT, filename + '.jpg')
        label_path = tmpdir.join(LABEL_ROOT, filename + '.png')

        image = np.random.randint(255, size=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        image = Image.fromarray(image.astype(np.uint8))
        label = np.random.randint(10, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        label = Image.fromarray(label.astype(np.uint8))

        # Convert path from py.path.local to str.
        image.save(str(image_path))
        label.save(str(label_path))

        val_image_list.append(image_path)
        val_label_list.append(label_path)

    data_list['val'] = {
        'image_list': val_image_list,
        'label_list': val_label_list
    }

    root_dir_path = tmpdir.join()
    return root_dir_path, data_list


def test_voc_get_file_path(tmpdir):
    """Test it can get file paths from cityscapes like data structure.
    """
    input_dir, gt_data_list = _create_sample_voc_structure(tmpdir)
    output_dir = input_dir
    data_list = get_file_path(input_dir, TRAIN_FILENAMES, VAL_FILENAMES)
    for cat1, cat2 in zip(data_list.values(), gt_data_list.values()):
        for list1, list2 in zip(cat1.values(), cat2.values()):
            # Do not care about orders.
            assert set(list1) == set(list2)


def test_id2trainid_tensor():
    """Test it can correctly convert id to trainId.
    """
    IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100
    with tf.Graph().as_default():
        id_tensor = tf.placeholder(tf.int64, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        trainid_tensor = id2trainid_tensor(id_tensor)
        with tf.Session() as sess:
            trainid = sess.run(
                trainid_tensor,
                feed_dict={
                    id_tensor: np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1)) * 8.
                })
        # Checks output size is correct.
        assert trainid.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        # Checks id is correctly converted to trainId.
        assert trainid[0, 0, 0] == 8


def test_trainid2color_tensor():
    """Test it can correctly convert trainId to color.
    """
    IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100
    with tf.Graph().as_default():
        id_tensor = tf.placeholder(tf.int64, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        color_tensor = trainid2color_tensor(id_tensor)
        with tf.Session() as sess:
            color = sess.run(
                color_tensor,
                feed_dict={
                    id_tensor: np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1)) * 2.
                })
        # Checks output size is correct.
        assert color.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        # Checks trainId is correctly converted to color.
        assert np.all(color[0, 0, :] == np.array((0, 128, 0)))
