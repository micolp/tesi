import numpy
from skimage import data, filters, io
from matplotlib import pyplot
import matplotlib.patches as patches


image = io.imread('images/Picture.png', as_gray=True)
io.imshow(image)
pyplot.show()

# Create figure and axes
fig, ax = pyplot.subplots(1)
# Display the image
ax.imshow(image)

grid = numpy.full((8, 8), False)

grid[2, 0] = True
grid[6, 7] = True

square_height = (image.shape[0])/grid.shape[0]
square_width = (image.shape[1])/grid.shape[1]

for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        x = j * square_width
        y = i * square_height
        if grid[i, j]:
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), square_height, square_width, linewidth=1, edgecolor='r', facecolor='r')
        else:
            rect = patches.Rectangle((x, y), square_height, square_width, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

edges = filters.sobel(image)
io.imshow(edges)
io.show()
pyplot.show()


















