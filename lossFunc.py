import numpy as np 

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
#one hot-encoded labels
class_targets = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]])

# first: going through each of the rows in softmax_outputs and finding value in each indices stated in targets
if len(class_targets.shape) == 1:
    #used if the list is just sparse and has only one row 
    correct_cofindences = softmax_outputs[range(len(softmax_outputs)), class_targets]
elif len(class_targets.shape) == 2:
    #used for one hot-encoded labels with multiple rows
    correct_cofindences = np.sum(softmax_outputs * class_targets, axis = 1)

#If probability number is close to 1, -log(p) is close to 0 then low loss
#If probability number is small, -log(p) is large then high loss
neg_log = -np.log(correct_cofindences)

#finding average loss through finding mean of the loss values we got form neg_log
average_loss = np.mean(neg_log)
print(average_loss)