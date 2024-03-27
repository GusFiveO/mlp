import numpy as np

class Layer:
	def __init__(self, input_shape, lenght, activation, weights_initializer = None):
		self.activation = activation
		self.weights_initializer = weights_initializer
		self.lenght = lenght
		self.input_shape = input_shape
		self.__init_weights(input_shape, weights_initializer)

	def __init_weights(self, input_shape, initializer):
		if initializer:
			self.weights = np.matrix(np.zeros(shape=(self.lenght, input_shape)))

	def __repr__(self) -> str:
		return (f"activation: {self.activation}\ninitializer: {self.weights_initializer}\nlenght: {self.lenght}\ninput_shape: {self.input_shape}\n" 
			+ ( f"weights: {self.weights.shape}\nweights content: {self.weights}" if self.weights_initializer is not None else ""))


class NeuralNetwork:

	def __init__(self, features, targets, epochs, learning_rate, layer_shapes_list, initializer):
		self.features = features
		self.targets = targets
		self.epochs = epochs
		self.learing_rate = learning_rate
		self.__normalize_features()
		input_shape = features.shape[1]
		output_shape = targets.unique().shape[0]
		self.__init_layers(input_shape, output_shape, layer_shapes_list, 'sigmoid', initializer)

	def __init_layers(self, input_shape, output_shape, shapes_list, activation, initializer):
		self.layers = list()
		self.layers.append(Layer(1, input_shape, activation))
		for i, layer_shape in enumerate(shapes_list):
			self.layers.append(Layer(input_shape, layer_shape, activation, initializer))
			input_shape = layer_shape
		self.layers.append(Layer(input_shape, output_shape, 'softmax', initializer))

	def __normalize_features(self):
		self.features = (self.features - self.features.max() / self.features.min()) * 2 - 1
	
	def __repr__(self) -> str:
		repr_string = f"features: {self.features.shape}\ntargets: {self.targets.shape}\nepochs: {self.epochs}\nlearning_rate: {self.learing_rate}\n"
		for i, layer in enumerate(self.layers):
			repr_string += f"\nlayer n{i}: " + repr(layer) 
		return repr_string

	def forward_propagation(self):
		for index, row in self.features.iterrows():
			print(np.array(row.values))

