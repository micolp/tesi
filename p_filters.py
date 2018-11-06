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
        description = str(self.__class__)
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
class Frangi(AbstractFilter):
    def __init__(self):
        self.filter = filters.frangi
        self.params = {"scale_range": (1, 10), "scale_step": 0.1, "beta1": 3, "beta2": 4, "black_ridges": False}
        # self.randomize()

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
class Gaussian(AbstractFilter):
    def __init__(self):
        self.filter = filters.gaussian
        self.params = {"sigma": 1, "output": None, "mode": '', "cval": 0,
                       "multichannel": None, "preserve_range": False, "truncate": 4.0}
        self.randomize()

    def apply(self, image):
        gaussian = self.filter(image, sigma=self.params['sigma'], output=self.params["output"], mode=self.params['mode'],
                               cval=self.params['cval'], multichannel=self.params['multichannel'],
                               preserve_range=self.params['preserve_range'], truncate=self.params['truncate'])
        return gaussian

    def randomize(self):
        self.params['sigma'] = random.uniform(0.4, 10.0)
        self.params['mode'] = random.choice(['reflect', 'constant', 'nearest', 'mirror', 'wrap'])
        if self.params['mode'] == 'constant':
            self.params['cval'] = random.uniform(-1.0, 1.0)


# ----------------------------------------------------------------------------------------------------------------------
class Laplacian(AbstractFilter):
    def __init__(self):
        self.filter = filters.laplace
        self.params = {"ksize": 3, "mask": None}
        # self.randomize()

    def apply(self, image):
        laplacian = self.filter(image, ksize=self.params['ksize'], mask=self.params['mask'])
        return laplacian

    def randomize(self):
        self.params['ksize'] = random.randint(3, 5)


# ----------------------------------------------------------------------------------------------------------------------
class Hessian(AbstractFilter):
    def __init__(self):
        self.filter = filters.hessian
        self.params = {'scale_range': (1, 10), 'scale_step': 2, 'beta1': 0.5, 'beta2': 15 }
        # self.randomize()

    def apply(self, image):
        hessian = self.filter(image, scale_range=self.params['scale_range'], scale_step=self.params['scale_step'],
                              beta1=self.params['beta1'], beta2=self.params['beta2'])
        return hessian

    def randomize(self):
        self.params['scale_range'] = (random.uniform(0.9, 1.0), random.uniform(9.9, 10.0))
        self.params['scale_step'] = random.uniform(0.0, 3.0)
        self.params['beta1'] = random.uniform(-0.5, 0.5)
        self.params['beta2'] = random.uniform(14.8, 15.0)


# ----------------------------------------------------------------------------------------------------------------------
# FEATURE
# ----------------------------------------------------------------------------------------------------------------------
class Canny(AbstractFilter):
    def __init__(self):
        self.filter = feature.canny
        self.params = {'sigma': 1.0, 'low_threshold': None, 'high_threshold': None,
                       'mask': None, 'use_quantiles': False}
        self.randomize()

    def apply(self, image):
        canny = self.filter(image, self.params['sigma'], self.params['low_threshold'], self.params['high_threshold'],
                            mask=self.params['mask'], use_quantiles=self.params['use_quantiles'])
        return canny.astype(int)

    def randomize(self):
        self.params['sigma'] = random.uniform(0.4, 10.0)
        # self.params['low_threshold'] = random.uniform()
        # self.params['high_threshold'] = random.uniform()


# ----------------------------------------------------------------------------------------------------------------------
# MORPHOLOGY
# ----------------------------------------------------------------------------------------------------------------------
class Erode(AbstractFilter):
    def __init__(self):
        self.filter = morphology.erosion
        self.params = {'selem': None, 'out': None, 'shift_x': False, 'shift_y': False}

    def apply(self, image):
        erode = self.filter(image, selem=self.params['selem'], out=self.params['out'],
                            shift_x=self.params['shift_x'], shift_y=self.params['shift_y'])
        return erode

    def randomize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Dilate(AbstractFilter):
    def __init__(self):
        self.filter = morphology.dilation
        self.params = {'selem': None, 'out': None, 'shift_x': False, 'shift_y': False}

    def apply(self, image):
        dilate = self.filter(image, selem=self.params['selem'], out=self.params['out'],
                             shift_x=self.params['shift_x'], shift_y=self.params['shift_y'])
        return dilate

    def randomize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Open(AbstractFilter):
    def __init__(self):
        self.filter = morphology.opening
        self.params = {'selem': None, 'out': None}

    def apply(self, image):
        open = self.filter(image, selem=self.params['selem'], out=self.params['out'])
        return open

    def randomize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Close(AbstractFilter):
    def __init__(self):
        self.filter = morphology.closing
        self.params = {'selem': None, 'out': None}

    def apply(self, image):
        close = self.filter(image, selem=self.params['selem'], out=self.params['out'])
        return close

    def randomize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Skeleton(AbstractFilter):
    def __init__(self):
        self.filter = morphology.skeletonize
        self.params = {}

    def apply(self, image):
        skeleton = self.filter(image)
        return skeleton.astype(int)

    def randomize(self):
        pass


# ----------------------------------------------------------------------------------------------------------------------
class Thin(AbstractFilter):
    def __init__(self):
        self.filter = morphology.thin
        self.params = {'max_iter': None}
        # self.randomize()

    def apply(self, image):
        thin = self.filter(image, max_iter=self.params['max_iter'])
        return thin.astype(int)

    def randomize(self):
        self.params['max_iter'] = random.randint(1, 10)


# ----------------------------------------------------------------------------------------------------------------------
# SETS
# ----------------------------------------------------------------------------------------------------------------------
edge_detector_set = (Sobel, Roberts, Prewitt, Scharr, Canny)
threshold_set = (ThresholdLocal, ThresholdGlobal)
morphology_set = (Erode, Dilate, Open, Close, Skeleton)#, Thin
misc_set = (Hessian, Laplacian, Gaussian)#, Frangi

category_set = (edge_detector_set, threshold_set, morphology_set, misc_set)

