class DummyNN(object):

    def __init__(self, num_inputs, learning_rate, momentum, src_file=None):

        self.num_inputs = num_inputs
        self.learning_rate = learning_rate

    def write_to_file(self, dst_file):

        pass

    def train_with_datapoint(self, inputs, target):

        pass

    def evaluate(self, inputs):

        return 1
