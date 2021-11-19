import numpy as np

a = np.array([[1,2]], dtype=np.float32)
b = np.array([[3,4]], dtype=np.float32)
c = np.array([[5,6]], dtype=np.float32)

l = [a, b, c]
l2 = l.copy()

for idx, x in enumerate(l):
    if idx == 0:
        try:

            l2.remove(x)
        except ValueError:
            print("failed to remove")
for idx, x in enumerate(l):
    if idx == 0:
        try:

            l2.remove(x)
        except ValueError:
            print("failed to remove")
print(l2)