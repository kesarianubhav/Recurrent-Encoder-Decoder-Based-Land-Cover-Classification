from time import time
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
import numpy as np
from utils import getFilesInDir, create_labels
from livePlots import PlotLearning
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def get_class(a):
    max = -99999
    idx = -99999
    for i in range(0, a.shape[0]):
        if a[i][0] > max:
            idx = i + 1
            max = a[i][0]
    return idx


def img_to_tensor(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    tensor = img_to_array(img)
    # print(tensor.shape)
    tensor = np.expand_dims(tensor, axis=0)
    # tensor = preprocess_input(tensor)
    print("Image """ + str(image_path) +
          " "" converted to tensor with shape " + str(tensor.shape))
    return tensor


def preprocess_input_rnn(tensor, n_time_steps, n_inputs):
    # time_steps = tensor.shape[3]
    print("preprocessing input for rnn")
    tensor = tensor.reshape((n_time_steps, n_inputs))
    return tensor

# input shape = (Batch Size , Number of Time Steps ,No. of inputs )


class RnnClassifier(object):

    def __init__(self, input_size, num_outputs=21):
        self.n_time_steps = input_size[0]
        self.n_units = 128
        self.n_inputs = input_size[1]
        self.n_classes = num_outputs
        self.batch_size = 128  # Size of each batch
        self.n_epochs = 50
        self._data_loaded = False
        self._trained = False

    def __create_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(self.n_units, input_shape=(self.n_time_steps, self.n_inputs)))
        self.model.add(Dense(self.n_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop', metrics=['accuracy'])

        print(self.model.summary())

    def train_model(self, input_tensors, output_tensors):
        print("Training started\n")
        self.__create_model()
        if self._trained == False:
            x_train = input_tensors
            x_train = np.array(x_train).reshape((-1,
                                                 self.n_time_steps, self.n_inputs))
            y_train = output_tensors
            y_train = np.array(y_train).reshape((-1, self.n_classes))
            plot_losses = PlotLearning()
            self.hist = self.model.fit(x_train, y_train,
                                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

            self._trained = True
            t = time()
            self.model.save("saved_model/" + str(t))

    def evaluate(self, input_tensors, output_tensors, model=None):
        # if self._trained == False and model == None:
        #     print("[!] Error:")
        #     sys.exit(0)

        x_test = input_tensors
        x_test = np.array(x_test).reshape((-1,
                                           self.n_time_steps, self.n_inputs))
        y_test = output_tensors
        y_test = np.array(y_test).reshape((-1, self.n_classes))
        self.model.fit(x_test, y_test,
                       batch_size=self.batch_size, epochs=self.n_epochs, shuffle=False)

        # model = load_model(model) if model else self.model
        test_loss = self.model.evaluate(x_test, y_test)
        print(test_loss)

    # def lossPlot(self, loss, epoch):

    def predict(self, input_tensors):
        outputs = []

        for i in input_tensors:
            i = np.array(i).reshape((-1,
                                     self.n_time_steps, self.n_inputs))
            outputs.append(self.model.predict_classes(i))

        return outputs

    def parameterPlot(self, parameter):

        for i in parameter:
            y_values = self.hist.history[str(i)]
            x_values = [i for i in range(1, self.n_epochs + 1)]
            plt.plot(x_values, y_values, label='line_' + str(parameter))
        plt.legend()
        plt.savefig("graphs/plot" + str(int(time())) + ".png", format='png')
        plt.show()
        return

if __name__ == '__main__':
    r = RnnClassifier(input_size=(224, 224))

    # r.train_model(input)
    r = np.random.randn(5, 1)
    print(r)
    print(get_class(r))
