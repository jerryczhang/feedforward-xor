import numpy as np
import math


class NeuralNetwork:
    """Neural Network with 2 Hidden Layers"""

    def __init__(self):
        """Initialize number of nodes for each layer, and set each value to random array"""

        self.input_layer_nodes = 5
        self.hidden_layer1_nodes = 5
        self.hidden_layer2_nodes = 5
        self.output_layer_nodes = 1

        self.w1 = np.random.randn(self.input_layer_nodes, self.hidden_layer1_nodes)
        self.b1 = np.random.randn(1, self.hidden_layer1_nodes)
        self.w2 = np.random.randn(self.hidden_layer1_nodes, self.hidden_layer2_nodes)
        self.b2 = np.random.randn(1, self.hidden_layer2_nodes)
        self.w3 = np.random.randn(self.hidden_layer2_nodes, self.output_layer_nodes)
        self.b3 = np.random.randn(1, self.output_layer_nodes)

        self.sigmoid = np.vectorize(self.sigmoid)
        self.sigmoid_prime = np.vectorize(self.sigmoid_prime)

        self.error_scalar = 1

    def forward_propagate(self):
        """Propagate an input forward through the net"""

        self.z2 = np.dot(self.x, self.w1) + self.b1
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2) + self.b2
        self.a3 = self.sigmoid(self.z3)
        self.z4 = np.dot(self.a3, self.w3) + self.b3
        self.yhat = self.sigmoid(self.z4)

        return self.yhat

    def calculate_gradients(self):
        """Calculate the gradients for each weight and bias, moving backward"""

        gradients = {}
        self.cost = np.sum((self.yhat - self.y) ** 2) / self.yhat.shape[0]
        dc_dyhat = 2 * (self.yhat - self.y)
        dyhat_dz4 = self.sigmoid_prime(self.z4)
        dc_dz4 = np.multiply(dc_dyhat, dyhat_dz4)
        dz4_dw3 = self.a3.T
        gradients['dc_dw3'] = np.dot(dz4_dw3, dc_dz4)
        gradients['dc_db3'] = np.sum(dc_dz4, axis=0)

        dz4_da3 = self.w3.T
        da3_dz3 = self.sigmoid_prime(self.z3)
        dc_dz3 = np.multiply(np.dot(dc_dz4, dz4_da3), da3_dz3)
        dz3_dw2 = self.a2.T
        gradients['dc_dw2'] = np.dot(dz3_dw2, dc_dz3)
        gradients['dc_db2'] = np.sum(dc_dz3, axis=0)

        dz3_da2 = self.w2.T
        da2_dz2 = self.sigmoid_prime(self.z2)
        dc_dz2 = np.multiply(np.dot(dc_dz3, dz3_da2), da2_dz2)
        dz2_dw1 = self.x.T
        gradients['dc_dw1'] = np.dot(dz2_dw1, dc_dz2)
        gradients['dc_db1'] = np.sum(dc_dz2, axis=0)

        return gradients

    def back_propagate(self):
        """Change each weight and bias by its calculated gradient"""

        gradients = self.calculate_gradients()

        self.w3 -= gradients['dc_dw3'] * self.error_scalar
        self.b3 -= gradients['dc_db3'] * self.error_scalar

        self.w2 -= gradients['dc_dw2'] * self.error_scalar
        self.b2 -= gradients['dc_db2'] * self.error_scalar

        self.w1 -= gradients['dc_dw1'] * self.error_scalar
        self.b1 -= gradients['dc_db1'] * self.error_scalar

    def train_network(self, epochs, training_data, training_labels, training_method='batch'):
        if training_method == 'batch':
            total_cost = 0
            for epoch in range(1, epochs + 1):
                self.x = training_data
                self.y = training_labels
                self.forward_propagate()
                self.back_propagate()
                total_cost += self.cost

                percent_complete = 100 * epoch / epochs
                if percent_complete % 5 == 0:
                    print('Epoch:', epoch, 'Percent Complete: %' + str(percent_complete), 'Average Cost:', total_cost / epoch)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return math.exp(x) / (math.exp(x) + 1) ** 2

    def print_net(self):
        """Print values for weights, biases, and node values"""

        print("\nWeight Layer 1:")
        print(self.w1)
        print("\nHidden Layer 1 Biases")
        print(self.b1)
        print("\nWeight Layer 2:")
        print(self.w2)
        print("\nHidden Layer 2 Biases:")
        print(self.b2)
        print("\nWeight Layer 3:")
        print(self.w3)
        print("\nHidden Layer 3 Biases")
        print(self.b3)

NN = NeuralNetwork()
training_data = np.array([
    # values that output 0
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1],

    # values that output 1
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
])

training_labels = np.array([[0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1]])
NN.train_network(1000, training_data, training_labels)
NN.print_net()

while True:
    input_data = []
    a = input("Input: ")
    for letter in a:
        input_data.append(int(letter))
    NN.x = np.array(input_data)
    print(NN.x)
    NN.forward_propagate()
    print(NN.yhat)


