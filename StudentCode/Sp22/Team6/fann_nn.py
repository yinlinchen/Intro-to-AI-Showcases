from fann2 import libfann


class FANN(object):

    def __init__(self, tot_inputs, learning_rate, momentum, src_file=None):

        self.nn = libfann.neural_net()

        if src_file is not None:
            self.nn.create_from_file(src_file)
        else:
            self.nn.create_standard_array([tot_inputs, 20, 1])

            #setting activation function of hidden and output of nn
            self.nn.set_activation_function_output(libfann.LINEAR)
            self.nn.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)


        #setting learning rate and momentum of nn
        self.nn.set_learning_rate(learning_rate)
        self.nn.set_learning_momentum(momentum)
        #incremental training algorithm
        self.nn.set_training_algorithm(libfann.TRAIN_INCREMENTAL)

        self.num_inputs = tot_inputs
        self.learning_rate = learning_rate
        self.momentum = momentum

    def write_to_file(self, dst_file):
        self.nn.save(dst_file)

    def train_with_datapoint(self, inputs, target):

        self.nn.train(inputs, [target])

    def evaluate(self, inputs):
        return self.nn.run(inputs)[0]
