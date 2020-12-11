from keras.datasets import mnist
from matplotlib import pyplot
import random

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[random.randrange(0, len(train_X)-1)], cmap=pyplot.get_cmap('gray'))
    
pyplot.show()