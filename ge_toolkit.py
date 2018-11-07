from random import choice

import numpy as np
from skimage import io

import p_filters
import ge_config as cfg


def load_training_set_kittens():
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
    oracle = np.zeros(shape=image.shape).astype(bool)
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


def load_training_set_scratches():
    training_set = []
    image = io.imread("images/02A_orig.bmp")
    oracle = np.zeros(shape=image.shape).astype(bool)
    coordinates = []
    try:
        with open("images/02A_orig.oracle") as oracle_02A:
            coordinates = oracle_02A.readlines()
    except FileNotFoundError as e:
        print(str(e))
    for s in coordinates:
        x = s.split(",")[0]
        y = s.split(",")[1]
        x = int(x)
        y = int(y)
        oracle[y-32:y+32, x-32:x+32] = True

    scratches = {
        'image': image,
        'oracle': oracle
    }
    training_set.append(scratches)
    return training_set


def load_training_set_scratches_short(square_size, i, j):
    training_set = []
    large_training_set = load_training_set_scratches()
    image = large_training_set[0]["image"]
    oracle = large_training_set[0]["oracle"]
    image = image[i:i+square_size, j:j+square_size]
    oracle = oracle[i:i+square_size, j:j+square_size]

    small_scratches = {
        'image': image,
        'oracle': oracle
    }
    training_set.append(small_scratches)
    return training_set


def load_training_set_final(block_size):
    training_set = []
    large_training_set = load_training_set_scratches()
    image = large_training_set[0]["image"]
    oracle = large_training_set[0]["oracle"]
    width = image.shape[1]
    height = image.shape[0]
    cols = int(width/block_size)
    rows = int(height/block_size)
    for i in range(10): #(rows):
        for j in range(10): #(cols):
            next_image = image[i:i + block_size, j:j + block_size]
            next_oracle = oracle[i:i + block_size, j:j + block_size]

            next_example = {
                'image': next_image,
                'oracle': next_oracle
            }
            training_set.append(next_example)
    return training_set


def get_random_filter():
    random_category = choice(p_filters.category_set)
    random_filter_class = choice(random_category)
    return random_filter_class()

