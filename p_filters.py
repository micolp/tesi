import random
from skimage import filters, morphology, feature


# ----------------------------------------------------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self):
        self.filters_list = []

    def process(self, image, normalize=False, verbose=False):
        for filter in self.filters_list:
            try:
                image = filter.apply(image)
            except RuntimeError as error:
                if verbose:
                    print(filter.get_description())
                    print("Error: " + str(error))
            except ValueError as error:
                if verbose:
                    print(filter.get_description())
                    print("Error: " + str(error))
        if normalize:
            image -= image.min()
            max_image = image.max()
            if max_image != 0:
                try:
                    image = image * (1.0 / max_image)
                except TypeError as error:
                    print("Error: " + str(error))
                    print(type(image))
                    print(max_image)
                    print(image.min())
                    print(type(max_image)) # int32
                    print(type(1.0 / max_image)) # float64
        return image

    def add_filter(self, filter, index=None):
        if index is None:
            self.filters_list.append(filter)
        else:
            self.filters_list.insert(index, filter)

    def remove_filter(self, index):
        del self.filters_list[index]

    def get_length(self):
        return len(self.filters_list)

    # index: da quale indice partire - offset: lunghezza subpipe
    def get_subpipeline(self, index, offset=None):
        result = Pipeline()
        if offset:
            result.filters_list = self.filters_list[index:index+offset]
        else:
            result.filters_list = self.filters_list[index:]
        # ritorna pipeline da: index, di lunghezza offset
        return result

    def get_description(self):
        description = ""
        for filter in self.filters_list:
            description += filter.get_description() + "\n"
            description += "--------------------\n"
        return description

    # override di add (+) per istanze di Pipeline
    def __add__(self, pipeline):
        result = Pipeline()
        result.filters_list = self.filters_list + pipeline.filters_list
        return result


# ----------------------------------------------------------------------------------------------------------------------
# ABSTRACT CLASS FILTERS
# ----------------------------------------------------------------------------------------------------------------------
class AbstractFilter:
    def get_description(self):
        description = str(self.__class__) + "\n"
        for param, value in self.params.items():
            description += "\n" + param + ": " + str(value)
        return description


# ----------------------------------------------------------------------------------------------------------------------
# FILTERS
# ----------------------------------------------------------------------------------------------------------------------
class ThresholdGlobal(AbstractFilter):
    def __init__(self):
        self.filters = {'mean': filters.threshold_mean, 'minimum': filters.threshold_minimum,
                        'otsu': filters.threshold_otsu}
        self.params = {'chosen_filter': ''}
        self.randomize()

    def randomize(self):
        self.params['chosen_filter'] = random.choice(list(self.filters.keys()))

    def apply(self, image):
        threshold_g = (self.filters[self.params['chosen_filter']])(image)
        return (image > threshold_g).astype(int)


# ----------------------------------------------------------------------------------------------------------------------
class ThresholdLocal(AbstractFilter):
    def __init__(self):
        self.filter = filters.threshold_local
        self.params = {"block_size": 0, "method": '', "offset": 0, "mode": '', "param": None, "cval": 0}
        self.randomize()

    def apply(self, image):
        threshold_l = self.filter(image, self.params["block_size"], method=self.params["method"],
                                  offset=self.params["offset"], mode=self.params["mode"],
                                  param=self.params["param"], cval=self.params["cval"])
        return (image > threshold_l).astype(int)

    def randomize(self):
        self.params["block_size"] = random.randint(3, 50)
        if self.params["block_size"] % 2 == 0:
            self.params["block_size"] += 1
        self.params["method"] = random.choice(['gaussian', 'mean', 'median'])
        if self.params["method"] == 'gaussian':
            self.params["cval"] = random.randint(5, 50)
        self.params["offset"] = random.uniform(-0.2, 0.2)
        self.params["mode"] = random.choice(['reflect', 'constant', 'nearest', 'mirror', 'wrap'])
        if self.params["mode"] == 'constant':
            self.params["cval"] = random.uniform(-1.0, 1.0)


# ----------------------------------------------------------------------------------------------------------------------
class Sobel(AbstractFilter):
    def __init__(self):
        self.filter = filters.sobel
        self.params = {}
        self.randomize()

    def apply(self, image):
        return self.filter(image)

    def randomize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Roberts(AbstractFilter):
    def __init__(self):
        self.filters = {'roberts': filters.roberts, 'roberts_neg_diag': filters.roberts_neg_diag,
                        'roberts_pos_diag': filters.roberts_pos_diag}
        self.params = {'chosen_filter': ''}
        self.randomize()

    def randomize(self):
        self.params['chosen_filter'] = random.choice(list(self.filters.keys()))

    def apply(self, image):
        roberts = (self.filters[self.params['chosen_filter']])(image)
        return roberts


# ----------------------------------------------------------------------------------------------------------------------
class Scharr(AbstractFilter):
    def __init__(self):
        self.filters = {'scharr': filters.scharr, 'scharr_v': filters.scharr_v, 'scharr_h': filters.scharr_h}
        self.params = {'chosen_filter': ''}
        self.randomize()

    def randomize(self):
        self.params['chosen_filter'] = random.choice(list(self.filters.keys()))

    def apply(self, image):
        scharr = (self.filters[self.params['chosen_filter']])(image)
        return scharr


# ----------------------------------------------------------------------------------------------------------------------
class Prewitt(AbstractFilter):
    def __init__(self):
        self.filters = {'prewitt': filters.prewitt, 'prewitt_v': filters.prewitt_v, 'prewitt_h': filters.prewitt_h}
        self.params = {'chosen_filter': ''}
        self.randomize()

    def randomize(self):
        self.params['chosen_filter'] = random.choice(list(self.filters.keys()))

    def apply(self, image):
        prewitt = (self.filters[self.params['chosen_filter']])(image)
        return prewitt


# ----------------------------------------------------------------------------------------------------------------------
class Frangi:
    def __init__(self):
        self.filter = filters.frangi
        self.params = {"scale_range": (1, 10), "scale_step": 0.1, "beta1": 3, "beta2": 4, "black_ridges": False}

    def apply(self, image):
        frangi = self.filter(image, scale_range=self.params['scale_range'], scale_step=self.params['scale_step'],
                 beta1=self.params['beta1'], beta2=self.params['beta2'], black_ridges=self.params['black_ridges'])
        return frangi

    def randomize(self):
        self.params['scale_range'] = (random.uniform(0.9, 1.0), random.uniform(9.9, 10.0))
        self.params['scale_step'] = random.uniform(0.0, 3.0)
        self.params['beta1'] = random.uniform(-0.5, 5)
        self.params['beta2'] = random.uniform(-0.5, 5)
        self.params['black_ridges'] = random.choice([True, False])


# ----------------------------------------------------------------------------------------------------------------------
class Gaussian:
    def __init__(self):
        self.filter = filters.gaussian
        self.sigma = 1
        self.output = None
        self.mode = ''
        self.cval = 0
        self.multichannel = None
        self.preserve_range = False
        self.truncate = 4.0
        self.randomize()

    def apply(self, image):
        gaussian = self.filter(image, sigma=self.sigma, output=self.output, mode=self.mode, cval=self.cval,
                               multichannel=self.multichannel, preserve_range=self.preserve_range, truncate=self.truncate)
        return gaussian

    def randomize(self):
        self.sigma = random.uniform(0.4, 10.0)
        self.mode = random.choice(['reflect', 'constant', 'nearest', 'mirror', 'wrap'])
        if self.mode == 'constant':
            self.cval = random.uniform(-1.0, 1.0)

    def get_description(self):
        return "This is a Gaussian Filter with sigma:" + str(self.sigma) + ", mode:" + \
                self.mode + ", cval:" + str(self.cval)


# ----------------------------------------------------------------------------------------------------------------------
class Laplacian:
    def __init__(self):
        self.filter = filters.laplace
        self.ksize = 3
        self.mask = None

    def apply(self, image):
        laplacian = self.filter(image, ksize=self.ksize, mask=None)
        return laplacian

    def randomize(self):
        self.ksize = random.randint(1, 10)

    def get_description(self):
        return "This is a Laplace Filter with ksize:" + str(self.ksize)


# ----------------------------------------------------------------------------------------------------------------------
class Hessian:
    def __init__(self):
        self.filter = filters.hessian
        self.scale_range = (1, 10)
        self.scale_step = 2
        self.beta1 = 0.5
        self.beta2 = 15
        self.randomize()

    def apply(self, image):
        hessian = self.filter(image, scale_range=self.scale_range, scale_step=self.scale_step,
                              beta1=self.beta1, beta2=self.beta2)
        return hessian

    def randomize(self):
        self.scale_range = (random.uniform(0.9, 1.0), random.uniform(9.9, 10.0))
        self.scale_step = random.uniform(0.0, 3.0)
        self.beta1 = random.uniform(-0.5, 0.5)
        self.beta2 = random.uniform(14.8, 15.0)

    def get_description(self):
        return "This is a Hessian Filter with scale range:" + str(self.scale_range) +\
               ", scale step:" + str(self.scale_step) + ", beta1:" + str(self.beta1) +\
               ", beta2:" + str(self.beta2)


# ----------------------------------------------------------------------------------------------------------------------
# FEATURE
# ----------------------------------------------------------------------------------------------------------------------
class Canny:
    def __init__(self):
        self.filter = feature.canny
        self.sigma = 1.0
        self.low_threshold = None
        self.high_threshold = None
        self.mask = None
        self.use_quantiles = False
        self.randomize()

    def apply(self, image):
        canny = self.filter(image, self.sigma, self.low_threshold, self.high_threshold,
                            mask=self.mask, use_quantiles=self.use_quantiles)
        return canny.astype(int)

    def randomize(self):
        self.sigma = random.uniform(0.4, 10.0)
        # self.low_threshold = random.uniform()
        # self.high_threshold = random.uniform()

    def get_description(self):
        return "This is a Canny Filter with sigma:" + str(self.sigma) + ", low threshold:" + \
                str(self.low_threshold) + ", high threshold:" + str(self.high_threshold)


# ----------------------------------------------------------------------------------------------------------------------
# MORPHOLOGY
# ----------------------------------------------------------------------------------------------------------------------
class Erode:
    def __init__(self):
        self.filter = morphology.erosion
        self.selem = None
        self.out = None
        self.shift_x = False
        self.shift_y = False

    def apply(self, image):
        erode = self.filter(image,
                            selem=self.selem,
                            out=self.out,
                            shift_x=self.shift_x,
                            shift_y=self.shift_y)
        return erode

    def randomize(self):
        pass

    def get_description(self):
        return "This is an Erosion Filter"


# ----------------------------------------------------------------------------------------------------------------------
class Dilate:
    def __init__(self):
        self.filter = morphology.dilation
        self.selem = None
        self.out = None
        self.shift_x = False
        self.shift_y = False

    def apply(self, image):
        dilate = self.filter(image,
                             selem=self.selem,
                             out=self.out,
                             shift_x=self.shift_x,
                             shift_y=self.shift_y)
        return dilate

    def randomize(self):
        pass

    def get_description(self):
        return "This is a Dilation Filter"


# ----------------------------------------------------------------------------------------------------------------------
class Open:
    def __init__(self):
        self.filter = morphology.opening
        self.selem = None
        self.out = None

    def apply(self, image):
        open = self.filter(image,
                           selem=self.selem,
                           out=self.out)
        return open

    def randomize(self):
        pass

    def get_description(self):
        return "This is an Opening Filter"


# ----------------------------------------------------------------------------------------------------------------------
class Close:
    def __init__(self):
        self.filter = morphology.closing
        self.selem = None
        self.out = None

    def apply(self, image):
        close = self.filter(image,
                            selem=self.selem,
                            out=self.out)
        return close

    def randomize(self):
        pass

    def get_description(self):
        return "This is a Closing Filter"


# ----------------------------------------------------------------------------------------------------------------------
class Skeleton:
    def __init__(self):
        self.filter = morphology.skeletonize

    def apply(self, image):
        skeleton = self.filter(image)
        return skeleton.astype(int)

    def randomize(self):
        pass

    def get_description(self):
        return "This is a Skeletonize Filter"


# ----------------------------------------------------------------------------------------------------------------------
class Thin:
    def __init__(self):
        self.filter = morphology.thin
        self.max_iter = None
        self.randomize()

    def apply(self, image):
        thin = self.filter(image, max_iter=self.max_iter)
        return thin.astype(int)

    def randomize(self):
        self.max_iter = random.randint(1, 10)

    def get_description(self):
        return "This is a Thin Filter"


# ----------------------------------------------------------------------------------------------------------------------
# SETS
# ----------------------------------------------------------------------------------------------------------------------
edge_detector_set = (Sobel, Roberts, Prewitt, Scharr, Canny)
threshold_set = (ThresholdLocal, ThresholdGlobal)
morphology_set = (Erode, Dilate, Open, Close, Skeleton, Thin)
misc_set = (Gaussian, Laplacian, Hessian, Frangi)

category_set = (edge_detector_set, threshold_set, morphology_set, misc_set)
