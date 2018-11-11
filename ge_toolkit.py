from random import choice

from skimage import io

import ge_config as cfg


def load_training_set_scratches():
    training_set = []
    image = io.imread("images/02A_orig.bmp")
    coordinates_scratches = []
    coordinates_no_scratches = []
    try:
        with open("images/02A_orig.oracle") as oracle_02A:
            coordinates_scratches = oracle_02A.readlines()
    except FileNotFoundError as e:
        print(str(e))
    try:
        with open("images/no_scratches.oracle") as oracle_no_scratches:
            coordinates_no_scratches = oracle_no_scratches.readlines()
    except FileNotFoundError as e:
        print(str(e))
    for s in coordinates_scratches:
        x = s.split(",")[0]
        y = s.split(",")[1]
        x = int(x)
        y = int(y)
        next_image = image[y - 32:y + 32, x - 32:x + 32]

        scratch = {
            'image': next_image,
            'oracle': True
        }
        training_set.append(scratch)

    for s in coordinates_no_scratches:
        x = s.split(",")[0]
        y = s.split(",")[1]
        x = int(x)
        y = int(y)
        next_image = image[y - 32:y + 32, x - 32:x + 32]

        no_scratch = {
            'image': next_image,
            'oracle': False
        }
        training_set.append(no_scratch)

    return training_set


training_set_list = {
                     'scratches': load_training_set_scratches
                    }


def get_random_filter():
    random_category = choice(cfg.category_set)
    random_filter_class = choice(random_category)

    return random_filter_class()


