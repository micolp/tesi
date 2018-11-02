import random
import __generator


n = random.randint(2, 10)
low = 0
high = 50
step = 1
filter = __generator.generate_matrix(n, low, high, step)
print(filter)






