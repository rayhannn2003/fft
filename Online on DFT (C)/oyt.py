import numpy as np

#np roll example
a = np.array([1, 2, 3, 4, 5])
print(np.roll(a, 2))  # Output: [4 5 1 2 3]
print(np.roll(a, -2)) # Output: [3 4 5 1 2]
print(np.roll(a, 0))  # Output: [1 2 3 4 5]
print(np.roll(a, 5))  # Output: [1 2 3 4 5]
print(np.roll(a, -4)) # Output: [2 3 4 5 1]
