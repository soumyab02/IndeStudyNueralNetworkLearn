import numpy as np

inputs = [  [1.0,2.0,3.0,2.5],
            [2.0,5.0,-1.0,2.0], 
            [-1.5,2.7,3.3,-0.8]  ]
#hidden layer1 is weights1 and biases1
weights1 = [ [0.2,0.8,-0.5,1],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87] ]
#hidden layer2 is weights2 and biases2
weights2 = [[0.1, -0.14, 0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13] ]
biases1 = [2.0, 3.0, 0.5]
biases2 = [-1, 2, -0.5]

layer_outputs1 = np.dot(inputs, np.array(weights1).T) + biases1
layer_outputs2 = np.dot(layer_outputs1, np.array(weights2).T) + biases2
print(layer_outputs2)


