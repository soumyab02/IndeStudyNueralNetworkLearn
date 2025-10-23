import numpy as np

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []

#finds which one is larger (0 or the input value)
outputs = np.maximum(0, inputs)
print(outputs)

#ReLU activation
class Activation_ReLU:
    #Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


'''
Simple Way to Understand Function and what it is doing
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
outputs = []
for i in inputs:
    if i > 0:
        outputs.append(i)
    else:
        outputs.append(0)
'''