from p_filters import category_set
from skimage import io
from time import clock, time

image = io.imread('images/Kitties.jpg', as_gray=True)

supported_input = ("float64", "int32")

perf_results = []
for i in range(1):
    for category in category_set:
        for filter_class in category:
            tic = clock()
            filter = filter_class()
            filter.apply(image)
            toc = clock()
            result = {
                'filter': filter,
                'time': toc-tic
            }
            perf_results.append(result)

perf_results.sort(key=lambda result: result['time'], reverse=True)
for result in perf_results:
    print('--------------------------------------')
    print('TIME:' + str(result['time']))
    print(result['filter'].get_description())
    print('--------------------------------------')
