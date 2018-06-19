# MNIST_Neural_Network_META

Simple fully connected neural network trained using gradient descent on the MNIST dataset. 
Sigmoid activation function.

## Meta
"Meta" neural networks train on the activation of the hidden layers of previously trained neural networks. 
META_NET_l256#59904#EPOCHS(24).nt is only trained on the hidden layers of the 3 other NNs and achieves an accuracy of 0.977 even though the best of the 3 "RESERVE_NN" only has an accuracy of 0.9721.

I plan to do further explore training on the activations of hidden layers of multiple trained NNs using TensorFlow. 
