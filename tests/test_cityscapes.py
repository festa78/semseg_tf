"""Test set for cityscapes classes.
"""

from PIL import Image
import numpy as np
import tensorflow as tf

from sss.data.cityscapes import get_cityscapes_file_path, id2trainid_tensor, trainid2color_tensor


def _create_sample_cityscapes_structure(tmpdir):
    """Creates dummy cityscapes like data structure.

    Returns
    -------
    root_dir_path : str
        Root path to the created data structure.
    data_list : dict
        Dummy data dictionary contains 'image_list' and 'label_list'.
    """
    # Constants.
    ROOTS = ['leftImg8bit', 'gtFine']
    SUBDIRS = ['aaa', 'bbb']
    DATA_CATEGORY = ['train', 'val', 'test']
    FILENAMES = ['test1', 'test2']
    IMAGE_WIDTH, IMAGE_HEIGHT = 100, 100

    np.random.seed(1234)

    data_list = {}
    for root in ROOTS:
        tmpdir.mkdir(root)
    for cat in DATA_CATEGORY:
        image_list = []
        label_list = []
        for root in ROOTS:
            tmpdir.mkdir(root, cat)
            for sub in SUBDIRS:
                tmpdir.mkdir(root, cat, sub)
                for filename in FILENAMES:
                    # Creates empty files.
                    if root == ROOTS[0]:
                        path = tmpdir.join(root, cat, sub, filename + '.png')
                        image = np.random.randint(
                            255, size=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
                        image = Image.fromarray(image.astype(np.uint8))
                        # Convert path from py.path.local to str.
                        image.save(str(path))
                        image_list.append(path)
                    else:
                        path = tmpdir.join(root, cat, sub,
                                           filename + '_labelIds.png')
                        label = np.random.randint(
                            10, size=(IMAGE_WIDTH, IMAGE_HEIGHT))
                        label = Image.fromarray(label.astype(np.uint8))
                        # Convert path from py.path.local to str.
                        label.save(str(path))
                        label_list.append(path)
        data_list[cat] = {'image_list': image_list, 'label_list': label_list}
    root_dir_path = tmpdir.join()
    return root_dir_path, data_list


def test_cityscapes_get_file_path(tmpdir):
    """Test it can get file paths from cityscapes like data structure.
    """
    input_dir, gt_data_list = _create_sample_cityscapes_structure(tmpdir)
    output_dir = input_dir
    data_list = get_cityscapes_file_path(input_dir)
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
        assert trainid[0, 0, 0] == 1


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
        assert np.all(color[0, 0, :] == np.array((70, 70, 70)))
