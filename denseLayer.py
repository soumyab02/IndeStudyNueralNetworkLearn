import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

#nnfs.init sets random seed to 0 and overrides the original dot product 
nnfs.init()

class Layer_Dense:
    #Layer initialization
    def __init__(self, n_inputs, n_nuerons):
        #initialize weights and biases 
        self.weights = 0.01 * np.random.randn(n_inputs, n_nuerons)
        self.biases = np.zeros((1,n_nuerons))
    #Forward pass
    def forward(self, inputs):
        #calculate output values from inputs, weights, and biases 
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
    #Backward Pass
    def backward(self, dvalues):
        #Gradiant on Parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        #Gradiant on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    #Forward Pass
    def forward(self, inputs):
        #remembers input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    #Backward Pass
    def backward(self, dvalues):
        #making copy of values first due to needing to modify original variable
        self.dinputs = dvalues.copy()
        #zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self,inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
    def backward(self, dvalues):
        #creating uninitialized array with meaningless values
        self.dinputs = np.empty_like(dvalues)
        #iterate sample-wise over pairs of the outputs and gradients 
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten the output array
            single_output = single_output.reshape(-1,1)
            #do calculation of the output 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #calculate sample-wise graident and add it to the array of the sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    #calculates the data and regularization of losses and given model output and ground truth values
    def calculate(self, output, y):
        #calculator sample losses
        sample_losses = self.forward(output, y)
        #calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    #y_true are the class targets
    def forward(self, y_pred, y_true):
        #probability numbers in a batch
        samples = len(y_pred)
        #prevent division by 0 and to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        #probabilities for target values - only if categorial labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        #probabilities for target values - only if hot-encoded labels 
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    #Backward Pass
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        #Number of labels in every sample
        #Use first sample to count them
        labels = len(dvalues[0])
        #If labels are sparse, turn them into one-hot encoder
        #Finds all of the samples in on of the rows which is y_true and labels is how many of the samples
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        #Calculate Gradient 
        self.dinputs = -y_true / dvalues
        #Normalize Gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():
    #creates activation and loss function objects 
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()
    #Forward Pass 
    def forward(self, inputs, y_true):
        #output layers activation function
        self.activation.forward(inputs)
        #Set the Output 
        self.output = self.activation.output 
        #Calculate and return loss value 
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        #number of samples
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #copy so we can safely modify
        self.dinputs = dvalues.copy()
        #calculate gradient 
        self.dinputs[range(samples), y_true] -= 1
        #Normalize gradient 
        self.dinputs = self.dinputs / samples
    
class Optimizer_SGD:
    #initialize optimizer and set settings
    #learning rate is 1 and is default for this optimizer 
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    #Call once before any parameter updates 
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    #Update parameters
    def update_params(self, layer):
        #If we use momentum
        if self.momentum:
            #If layer does not contaim momentum arrays, create them and fill with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            #Build weight updates with momentum and take previous updates and multiply by retain factor
            #and update with current gradients 
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            #Build bias updates 
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        #Vanilla SGD updates before momentum 
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        #update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    #initialize optimizer and set settings
    #learning rate is 1 and is default for this optimizer 
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = 1e-7
    #Call once before any parameter updates 
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))
    #Update parameters
    def update_params(self, layer):
        #If layer does not have cache array then create them filled with 0
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        #Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        #if layer doesn't contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        #update cache with squared current graidents 
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
    #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)  
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    #call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):
        #if layer doesn't contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        #update momentum with current gradients 
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        #get corrected momentum 
        weight_momentum_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        #update  cache with squared current gradients 
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        #get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
    #Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)  
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

#Create Dataset
X, y = spiral_data(samples=100, classes=3)
plt.figure(figsize=(3, 3)) 
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='brg')
plt.show()

#Create Model
dense1 = Layer_Dense(2, 16) #first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(16,3) #second dense layer, 3 inputs 3 outputs
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

#Train in loop
for epoch in range(10001):
     # 1. Create a Meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))

    # 2. Make Predictions on the Meshgrid
    # Flatten the meshgrid for input to the network
    Z_input = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass through the trained network
    dense1.forward(Z_input)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)

    # Get predictions (class labels)
    Z = np.argmax(dense2.output, axis=1)

    # Reshape the predictions back to the meshgrid shape
    Z = Z.reshape(xx.shape)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    #Calculate accuracy from output of activation2 and targets 
    #Calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 1000:
        print(f'epoch: {epoch}, ' +
            f'acc: {accuracy:.3f}, ' +
            f'loss: {loss:.3f}, ' +
            f'lr: {optimizer.current_learning_rate}')
        # 3. Plot the Contour
        plt.figure(figsize=(6, 6))
        plt.contourf(xx, yy, Z, cmap='brg', alpha=0.2)

        # 4. Overlay the Original Data
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='brg', edgecolors='k')
        plt.title('Neural Network Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    #Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    #Update weights and biases 
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
