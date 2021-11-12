# neuralnetwork-mnist

In this repository we are using a Deep Learning algorithm.
This algorithm is used to recognize hand-written digits. Using the MNIST Data Set which has a training set of 60_000 examples and a test set of 10_000 examples. Divided in two files, our "model.py" file helps creating and structuring our neural network. Our images size is 28x28 pixels so the first layer contains 784 (flattening, 28 * 28) neurons assigned to each pixel you may say while in our "main.py" we are declaring our model, hyperparameters, functions for optimizing the gradients and defining the loss, preprocessing our data using torch library, and computing the metrics. Looping through our dataset _epoch_ times we'll be saving our errors per epoch to see how our model is working out through training and validation.

As activation functions for the layers, we are using ReLU (r(x)=max(0,x)) and LogSoftmax for the final layer of the neural network. 
As loss function I used NLLLoss() function.
As optimizer I used the Stochastic Gradient Descent. (SGD)
