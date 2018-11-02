import random
from skimage import filters, io, morphology, feature

class Pipeline:
    def __init__(self):
        self.filters_list = []

    def process(self, image):
        for filter in self.filters_list:
            image = filter.apply(image)
        return image

    def add_filter(self, filter):
        self.filters_list.append(filter)


class Sobel:
    def __init__(self):
        self.filter = filters.sobel
        self.randomize()

    def apply(self, image):
        return self.filter(image)

    def randomize(self):
        pass

    def get_description(self):
        return "This is a Sobel filter"


class ThresholdGlobal:
    def __init__(self):
        self.filters = {'mean': filters.threshold_mean, 'minimum': filters.threshold_minimum,
                        'otsu': filters.threshold_otsu}
        self.chosen_filter = ""
        self.randomize()

    def randomize(self):
        self.chosen_filter = random.choice(list(self.filters.keys()))

    def apply(self, image):
        threshold = (self.filters[self.chosen_filter])(image)
        return image > threshold

    def get_description(self):
        return "This is a Threshold Global Filter using the " + self.chosen_filter + " filter"


class ThresholdLocal:
    def __init__(self):
        self.filter = filters.threshold_local
        self.block_size = 0
        self.method = ''
        self.offset = 0
        self.mode = ''
        self.param = None
        self.cval = 0
        self.randomize()

    def apply(self, image):
        threshold = self.filter(image, self.block_size, method=self.method, offset=self.offset, mode=self.mode,
                                param=self.param, cval=self.cval)
        return image > threshold

    def randomize(self):
        self.block_size = random.randint(3, 50)
        if self.block_size % 2 == 0:
            self.block_size += 1
        self.method = random.choice(['gaussian', 'mean', 'median'])
        if self.method == 'gaussian':
            self.param = random.randint(5, 50)
        self.offset = random.uniform(-0.2, 0.2)
        self.mode = random.choice(['reflect', 'constant', 'nearest', 'mirror', 'wrap'])
        if self.mode == 'constant':
            self.cval = random.uniform(-1.0, 1.0)

    def get_description(self):
        return "This is a Threshold Local Filter with block size:" + str(self.block_size) + ", method:" \
               + self.method + ", param:" + str(self.param) + ", offset:" + str(self.offset) + ", mode:" + \
               self.mode + ", cval:" + str(self.cval)


class Roberts:
    def __init__(self):
        self.filters = {'roberts': filters.roberts, 'roberts_neg_diag': filters.roberts_neg_diag,
                        'roberts_pos_diag': filters.roberts_pos_diag}
        self.chosen_filter = ""
        self.randomize()

    def randomize(self):
        self.chosen_filter = random.choice(list(self.filters.keys()))

    def apply(self, image):
        roberts = (self.filters[self.chosen_filter])(image)
        return roberts

    def get_description(self):
        return "This is a Roberts Filter using the " + self.chosen_filter + " filter"


class Scharr:
    def __init__(self):
        self.filters = {'scharr': filters.scharr, 'scharr_v': filters.scharr_v, 'scharr_h': filters.scharr_h}
        self.chosen_filter = ""
        self.randomize()

    def randomize(self):
        self.chosen_filter = random.choice(list(self.filters.keys()))

    def apply(self, image):
        scharr = (self.filters[self.chosen_filter])(image)
        return scharr

    def get_description(self):
        return "This is a Scharr Filter using the " + self.chosen_filter + " filter"


class Prewitt:
    def __init__(self):
        self.filters = {'prewitt': filters.prewitt, 'prewitt_v': filters.prewitt_v, 'prewitt_h': filters.prewitt_h}
        self.chosen_filter = ""
        self.randomize()

    def randomize(self):
        self.chosen_filter = random.choice(list(self.filters.keys()))

    def apply(self, image):
        prewitt = (self.filters[self.chosen_filter])(image)
        return prewitt

    def get_description(self):
        return "This is a Prewitt Filter using the " + self.chosen_filter + " filter"


class Frangi:
    def __init__(self):
        self.filter = filters.frangi
        self.scale_range = (1, 10)
        self.scale_step = 2
        self.beta1 = 0.5
        self.beta2 = 15
        self.black_ridges = True  # T detects black ridges, F white

    def apply(self, image):
        frangi = self.filter(image, scale_range=self.scale_range, scale_step=self.scale_step, beta1=self.beta1,
                             beta2=self.beta2, black_ridges=self.black_ridges)
        return frangi

    def randomize(self):
        self.scale_range = (random.uniform(0.9, 1.0), random.uniform(9.9, 10.0))
        self.scale_step = random.uniform(0.0, 3.0)
        self.beta1 = random.uniform(-0.5, 0.5)
        self.beta2 = random.uniform(-0.5, 0.5)
        self.black_ridges = random.choice([True, False])

    def get_description(self):
        return "This is a Frangi Filter with scale range:" + str(self.scale_range) + ", scale step:" + str(
            self.scale_step) \
               + ", beta1:" + str(self.beta1) + ", beta2:" + str(self.beta2) + ", black ridges: " + str(
            self.black_ridges)


edge_detector_set = (Sobel, Roberts, Prewitt, Scharr)
threshold_set = (ThresholdLocal, ThresholdGlobal)
misc_set = (Frangi, Frangi)
category_set = (edge_detector_set, threshold_set, misc_set)
