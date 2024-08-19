from algorithms.neural_network import NeuralNetwork

nn = NeuralNetwork(layer_dimensions=[3,2,1],learning_rate=0.001)
nn.initialize_parameters()

print(nn.__dict__)