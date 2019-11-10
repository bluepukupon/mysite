import os.path
import gzip
import pickle
import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.image import imread
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import cv2

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    
    return T


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))



def predict(x):
    W1 = np.load('weight_and_bias_W1.npy',None,True,True)
    W2 = np.load('weight_and_bias_W2.npy',None,True,True)
    b1 = np.load('weight_and_bias_b1.npy',None,True,True)
    b2 = np.load('weight_and_bias_b2.npy',None,True,True)
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)
    
    return y



# データの読み込み
img = cv2.imread('aaa.png')
img = cv2.imread('aaa.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray.jpg',gray)
counter = 0
for i in im:
    if i >= 80:
        im[counter] = i
    else:
        im[counter] = 0
    counter += 1

im = np.array(Image.open('gray.jpg'))
im = im.ravel()
result = predict(im)
plt.plot(result)
plt.show()

np.save('weight_and_bias_W1',network.params['W1'])
np.save('weight_and_bias_W2',network.params['W2'])
np.save('weight_and_bias_b1',network.params['b1'])
np.save('weight_and_bias_b2',network.params['b2'])

W1 = np.load('weight_and_bias_W1.npy',None,True,True)
W2 = np.load('weight_and_bias_W2.npy',None,True,True)
b1 = np.load('weight_and_bias_b1.npy',None,True,True)
b2 = np.load('weight_and_bias_b2.npy',None,True,True)


im = np.array(Image.open('gray.jpg'))
plt.imshow(im)
plt.show()



