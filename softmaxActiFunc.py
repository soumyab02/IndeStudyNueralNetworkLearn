import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
print('exponentiated values: ', exp_values)

norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentiated values: ', norm_values)
print('Sum of all normalized values: ', np.sum(norm_values))

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
