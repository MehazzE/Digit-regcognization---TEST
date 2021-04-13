import numpy as np
import cv2
import sys
import random
import network2 as nw2
import paint
import data_relative

n1 = nw2.Network([784,200,100,10])

biases, weights = data_relative.load_biases_weights()
n1.load_weights(biases, weights)

App = paint.App
window = paint.Window(n1)
window.show()
sys.exit(App.exec())