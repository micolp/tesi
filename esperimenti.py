import random
import p_filters
from skimage import filters, io, morphology, feature
from matplotlib import pyplot

image = io.imread('images/Picture.png', as_gray=True)
'''
s = p_filters.Frangi() #s è un'istanza della classe Frangi
print(s.get_description())
io.imshow(s.apply(image))
pyplot.show()
'''

random_category = random.choice(p_filters.category_set)
random_filter_class = random.choice(random_category)
random_filter = random_filter_class()

print(random_filter.get_description())
io.imshow(random_filter.apply(image))
pyplot.show()

