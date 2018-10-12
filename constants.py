
# Note - This file is just for the checking purpose .
# It itsn't used
# anywhere in the project

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
# parameters
n_time = 12
n_hidden_state = 10
n_classes = 21
img_rows = 224
img_columns = 224
epochs = 10
batch_size = 1

# n_steps is nothing but the rows in the images
# n_inputs is nothing but the number of columns in the images


# input_shape = (None , time_step(the rows) , number_of_inputs(the columns))
# number_of_outputs = number_of_classes
if __name__ == '__main__':
    mnist = input_data.read_data_sets("mnist", one_hot=True)
    x = mnist.train.images
    x = [i.reshape((-1, 28, 28)) for i in x]
    x = np.array(x).reshape((-1, 28, 28))
    print(x.shape)
    y = mnist.train.labels

    print(y.shape)
