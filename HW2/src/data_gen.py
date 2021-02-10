import numpy as np

class Data():
    def __init__(self):
        mnist_data = np.loadtxt('files/even_mnist.csv')

        x_train= mnist_data[3000:,:-1]
        y_train= mnist_data[3000:,-1]
        x_train= np.array(x_train, dtype= np.float32)
        y_train= np.array(y_train, dtype= np.float32)

        x_test= mnist_data[:3000,:-1]
        y_test= mnist_data[:3000, -1]
        x_test= np.array(x_test, dtype= np.float32)
        y_test= np.array(y_test, dtype= np.float32)

        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test
