x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

#multiply weights and input values 
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
#adding up all of the weighted inputs and biases 
z = xw0 + xw1 + xw2 + b

#ReLU activation function
y = max(z, 0)

#Backward Pass

#Derivative from the next layer 
dvalue = 1.0

#Derivative of ReLu and the chain rule 
drelu_dz = dvalue * (1. if z > 0 else 0.)

#Partial derivatives of the multiplication, the chain rule 
drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
drelu_dx1 = dvalue * (1. if z > 0 else 0.) * w[1]
drelu_dx2 = dvalue * (1. if z > 0 else 0.)* w[2]
drelu_dw0 = dvalue * (1. if z > 0 else 0.) * x[0]
drelu_dw1 = dvalue * (1. if z > 0 else 0.) * x[1]
drelu_dw2 = dvalue * (1. if z > 0 else 0.) * x[2]
drelu_db =  dvalue * (1. if z > 0 else 0.) * b

dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]

#multiply weights and input values 
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
#adding up all of the weighted inputs and biases 
z = xw0 + xw1 + xw2 + b

#ReLU activation function
y = max(z, 0)

print(y)

