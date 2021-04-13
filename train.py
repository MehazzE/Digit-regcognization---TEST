import numpy as np
import cv2
import sys
import random
import network2 as nw2
import data_relative

n1 = nw2.Network([784,200,100,10])

def train(nw, loop):
    td = data_relative.generate_training_data()
    biases, weights = data_relative.load_biases_weights()
    nw.load_weights(biases, weights)
    print("Training data:" + str(len(td)))
    for i in range(loop):
        batch_train = random.sample(td, 10)
        batch_test = td
        nw.update_mini_batch(batch_train, 0.01, 0.01, 20)
        print(str(i+1)+"/"+str(loop) + "; loss: "+str(round(nw.calculate_loss(batch_test), 2)), end="\r")
    print("LOSS: "+str(nw.calculate_loss(td)))
    data_relative.save_biases_weights(nw)

train(n1, 500)