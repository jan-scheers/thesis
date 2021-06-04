import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot

xsm = np.linspace(-2,2,1000)

pyplot.plot(xsm,tf.nn.relu(xsm))
pyplot.grid(True)
pyplot.xlim([-1,1])
pyplot.ylim([-1,1])
pyplot.figure()
pyplot.plot(xsm,tf.math.tanh(xsm))
pyplot.grid(True)
pyplot.xlim([-2,2])
pyplot.ylim([-1,1])
pyplot.show()
