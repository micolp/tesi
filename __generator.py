import numpy as np


def generate_matrix(n, low, high, step):
    matrix = np.random.choice([x for x in range(low, high, step)], n*n)
    matrix.resize(n, n)
    return matrix


'''
generates a random n matrix with 1 random value
filter = numpy.empty((n, n,))
filter[:] = numpy.random.randint(1, 1000)
print(filter)
'''


# def generate_pipe():
