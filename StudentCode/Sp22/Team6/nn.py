from fann_nn import FANN


class NN(object):


    #defining learning rate and momentum for the network
    learning_rate = 0.01
    momentum = 0.2

    def __init__(self, num_inputs, src_file=None):

        self.nn = FANN(num_inputs, NN.learning_rate, NN.momentum, src_file)

    def write_to_file(self, dst_file):

        self.nn.write_to_file(dst_file)

    def train_with_datapoint(self, inputs, target):

        self.nn.train_with_datapoint(inputs, target)

    def evaluate(self, inputs):

        return self.nn.evaluate(inputs)