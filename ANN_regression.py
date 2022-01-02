import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

X, y = sine_data()

plt.plot(X, y)
plt.show()
