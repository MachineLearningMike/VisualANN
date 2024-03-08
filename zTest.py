import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import itertools
import matplotlib.animation as animation
import matplotlib.lines as lines
from matplotlib.widgets import RadioButtons, Cursor, Button

def getRowsCols(n):
    rows = np.power(n, 0.5)
    cols = (n / rows)
    nRows, nCols = None, None
    if int(rows) < rows:
        if int(rows) * int(cols) >= rows * cols:
            nRows, nCols = int(rows), int(cols)
        else:
            if int(rows) * int(cols+1.) >= rows * cols:
                nRows, nCols = int(rows), int(cols+1.)
            else:
                nRows, nCols = int(rows+1.), int(cols+1.)
    else:
        nRows, nCols = int(rows), int(cols+.5)

    return nRows, nCols


for n in range(1,17):
    print(n, getRowsCols(n))


fig, axes = plt.subplots(nrows=2, ncols=3)

pass


import tensorflow as tf

for i in range(100000):
    a = tf.constant(i)
    b = tf.constant(i)
    c = tf.multiply(a, b)
    print(c.numpy())