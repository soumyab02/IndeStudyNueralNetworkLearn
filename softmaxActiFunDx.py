import numpy as np

softmax_output = [0.7, 0.1, 0.2]
softmax_output = np.array(softmax_output).reshape(-1,1)

print(softmax_output)

#computes an array using an input vector as the diagnol and the multiplying it by softmax_output 
print(np.diagflat(softmax_output))

#multiplication of softmax outputs iterating the j and k indices respectively
print(np.dot(softmax_output, softmax_output.T))

#next you need to subtract both of arays that have been made 
#which is following the derivative equation for softmax 
#this array solution is called Jacobian matrix
print(np.diagflat(softmax_output) - np.dot(softmax_output, softmax_output.T))

