from p_filters import category_set
from skimage import io

image = io.imread('images/image_short.png', as_gray=True)
print(str(type(image)))
print(image.dtype)

supported_input = ("float64", "int32")

for category in category_set:
    for filter_class in category:
        filter = filter_class()
        result = filter.apply(image)
        if image.shape != result.shape:
            print("--------------------------------------------------------------------------------------------------\n"
                  "PROBLEM! \n Image shape : " + str(image.shape[0]) + ", " + str(image.shape[1]) +
                  "\n Filtered image shape: " + str(image.shape[0]) + ", " + str(image.shape[1]) + "\n" +
                  "The filter says: \n" + filter.get_description())
        if str(type(result)) != "<class 'numpy.ndarray'>":
            print("--------------------------------------------------------------------------------------------------\n"
                  "PROBLEM! Return value isn't a ndarray but a " + str(type(result)) + ". "
                  "The filter says: \n" + filter.get_description())
        if str(result.dtype) not in supported_input:
            print("--------------------------------------------------------------------------------------------------\n"
                  "PROBLEM! Result type not supported. It was instead a " + str(result.dtype) + ".\n"
                  "The filter says: \n" + filter.get_description())
        print("------------------------------------------------------------------------------------------------------\n"
              + filter.get_description())






