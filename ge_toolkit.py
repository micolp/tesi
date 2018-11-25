from random import choice
from skimage import io
import ge_config as cfg
import numpy as np
from os import listdir


def load_training_set_scratches():
    training_set = []
    positives_image_files = listdir('images/positives')
    negatives_image_files = listdir('images/negatives')
    for positives_image_file in positives_image_files:
        tile = io.imread("images/positives/" + positives_image_file)
        training_set.append({
            'image': tile,
            'oracle': True
        })
    for negatives_image_file in negatives_image_files:
        tile = io.imread("images/negatives/" + negatives_image_file)
        training_set.append({
            'image': tile,
            'oracle': False
        })
    return training_set


training_set_list = {
                     'scratches': load_training_set_scratches
                    }


def filtered_tile_by_tile(pipeline, image):
    filtered_image = np.zeros(shape=image.shape)
    positive_negative_image = np.zeros(shape=image.shape)
    xs = [32 + i * 54 for i in range(37)]
    ys = [32 + i * 54 for i in range(37)]
    positives = []
    negatives = []
    for x in xs:
        for y in ys:
            tile = image[y - 32:y + 32, x - 32:x + 32]
            filtered_tile = pipeline.process(tile)
            if np.sum(filtered_tile)/filtered_tile.size >= 0.5:
                positives.append({'x': x, 'y': y})
                positive_negative_image[y - 32:y + 32, x - 32:x + 32] = 1
            else:
                negatives.append({'x': x, 'y': y})
                positive_negative_image[y - 32:y + 32, x - 32:x + 32] = 0
            filtered_image[y - 32:y + 32, x - 32:x + 32] = filtered_tile

    return filtered_image, positive_negative_image, positives, negatives


def get_random_filter():
    random_category = choice(cfg.category_set)
    random_filter_class = choice(random_category)

    return random_filter_class()


def log(message, verbose_level):
    if verbose_level <= cfg.verbose_level:
        print(message)


