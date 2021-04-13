import cv2
import numpy as np

def toArray(img, digit):
    y = np.zeros((10,1), dtype="int")
    y[digit][0] += 1
    x = []
    for r in range(0, len(img)):
        for c in img[r]:
            if c == 255:
                x.append([0])
            else:
                x.append([1])
    return (np.array(x), np.array(y))

def generate_training_data():
    training_data = []
    for d in range(0,10):
        i = 0
        while True:
            img = cv2.imread("./data/"+str(d)+"/"+str(i)+".png", 0)
            if img is None:
                break
            training_data.append(toArray(img, d))
            i += 1
    return training_data

def get_inspect_data():
    return toArray(cv2.imread("./data/test/inspect.png", 0),0)[0]

def save_biases_weights(nw):
    np.save("./data/biases_weights/biases.npy", nw.biases)
    np.save("./data/biases_weights/weights.npy", nw.weights)
    
def load_biases_weights():
    biases = np.load("./data/biases_weights/biases.npy", allow_pickle=True)
    weights = np.load("./data/biases_weights/weights.npy", allow_pickle=True)
    return biases, weights