from random import choice

import numpy as np
from skimage import io

import p_filters
import ge_config as cfg


def load_training_set():
    training_set = []
    image = io.imread(cfg.image_path, as_gray=True)

    # stiamo creando la nostra griglia valorizzata (matrice di False)
    oracle_bool = np.zeros(shape=(6, 10)).astype(bool)
    oracle_bool[2][3] = True
    oracle_bool[2][4] = True
    oracle_bool[2][6] = True
    oracle_bool[2][7] = True
    # oracle_bool = load_grid()
    # fino a qui
    # crea matrice di zeri (un'immagine tutta nera della stessa dim dell'immagine originale)
    oracle = np.zeros(shape=image.shape)
    square_height = (image.shape[0]) / oracle_bool.shape[0]
    square_width = (image.shape[1]) / oracle_bool.shape[1]
    # adatta griglia all'immagine di partenza, colorando di bianco (=1) dove il valore è true
    for i in range(oracle_bool.shape[0]):
        for j in range(oracle_bool.shape[1]):
            if oracle_bool[i, j]:
                oracle[int(i * square_height):int(i * square_height) + int(square_height),
                int(j * square_width):int(j * square_width) + int(square_width)] = 1

    kittens = {
        'image': image,
        'oracle': oracle
    }
    training_set.append(kittens)
    return training_set


def get_random_filter():
    random_category = choice(p_filters.category_set)
    random_filter_class = choice(random_category)
    return random_filter_class()
