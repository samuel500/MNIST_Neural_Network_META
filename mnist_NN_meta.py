import numpy as np
import json
import sys
import math
from time import time
import pickle
import os
import struct
from random import random
from random import choice

def def_sigmoid(x):
	if x>-64:
		return 1 / (1 + math.exp(-x))
	else:
		return 1

sigmoid = np.vectorize(lambda x: def_sigmoid(x))
deriv_sigm = np.vectorize(lambda x: x*(1-x))


def load_NN(name, test = False):

	with open(name, 'rb') as input:
		NN = pickle.load(input)
		if test:
			data = load_mnist(dataset = "testing", path = "")
			NN.test(data)

	return NN


def main():

	global data #For later tests
	l_rate = 0.4
	nb_epochs = 24

	#Load and test trained NN
	NNnames = [
		'RESERVE_NN_l64#50816#EPOCHS(24).nt',
		'RESERVE_NN_l32l16#25760#EPOCHS(24).nt',
		'RESERVE_NN_l64l48#53728#EPOCHS(24).nt',
		'META_NET_l256#59904#EPOCHS(24).nt'
	]



	try:
		for n in NNnames:
			print(n)
			lnn = load_NN(n, test=True)
		return
	except FileNotFoundError:
		print("NN to test not found")


	#Train NNs
	nns = [
		NeuralNet([{'name': 'layer1', 'neurNb': 64}]),
		NeuralNet([{'name': 'layer1', 'neurNb': 64}, {'name': 'layer2', 'neurNb': 48}]),
		NeuralNet([{'name': 'layer1', 'neurNb': 32}, {'name': 'layer2', 'neurNb': 16}]),
	]


	nameList = []

	for nn in nns:
		print(nn.h_layers)
		print("l_rate:", l_rate)
		#nn.__init__(nn.h_layers)
		l_rate = 0.5
		for i in range(nb_epochs):
			data = load_mnist(path = '')
			l_rate *= 0.95
			nn.training(data, l_rate)
			data = load_mnist(dataset = "testing", path = "")
			nn.test(data)


		data = load_mnist(dataset = "testing", path = "")
		nn.test(data)
		nn.saveNN('RESERVE_NN')


	#Train meta NN
	names = [
		'RESERVE_NN_l64#50816#EPOCHS(24).nt',
		'RESERVE_NN_l32l16#25760#EPOCHS(24).nt',
		'RESERVE_NN_l64l48#53728#EPOCHS(24).nt'
	]

	ni = []

	for n in names:
		lnn = load_NN(n, test=True)
		ni.append([lnn, list(range(1, n.count('l')+1))])

	newMetaNN = NeuralNet([{'name': 'layer1', 'neurNb': 256}], neural_inputs = ni, image_input=False)


	print(newMetaNN.h_layers)
	print("l_rate:", l_rate)
	#nn.__init__(nn.h_layers)
	for i in range(nb_epochs):

		data = load_mnist(path = '')
		newMetaNN.training(data, l_rate)
		data = load_mnist(dataset = "testing", path = "")
		newMetaNN.test(data)

	data = load_mnist(dataset = "testing", path = "")
	newMetaNN.test(data)
	newMetaNN.saveNN('META_NET')


class NeuralNet:

	#h_layers: structure of hidden layers
	#self_train: input and output are the same, attempt at compression (keep False)
	#neural_inputs: list of trained NN, activation of their hidden layers are used as input (ex: [[NN1, [1,2]],[NN2, [2]]]; Will take activation of hidden layer 1 and 2 of NN1 and activation of hidden layer 2 of NN2 as input)
	#image_input: if image is given as input of NN
	#outputs: output layer, train on outputs digits and "other",
	def __init__(self, h_layers =[{'name': 'layer1', 'neurNb': 16}, {'name': 'layer2', 'neurNb': 16}], self_train = False, neural_inputs = [], image_input = True, outputs = [0,1,2,3,4,5,6,7,8,9]):

		self.self_train = self_train
		self.neural_inputs = neural_inputs
		self.epochs = 0
		self.image_input = image_input
		self.outputs = list(outputs)
		print(self.outputs)

		if not self_train:
			self.outputL = [0]*len(outputs)
		else:
			self.outputL = list(range(784))
		self.layers_len = [x['neurNb'] for x in h_layers] + [len(self.outputL)]
		self.h_layers = h_layers
		nToAdd = 0 #number of inputs in addition to image
		if neural_inputs:
			for n in neural_inputs:
				print(n)
				for lNb in n[1]:
					nToAdd += len(n[0].b_form[lNb-1])
		else:
			nToAdd = 0

		if not image_input:
			self.inputL = np.array((nToAdd)*[0])
		else:
			self.inputL = np.array((784+nToAdd)*[0])

		#Creating weights matrix
		self.w_form = [[[0]*self.h_layers[0]['neurNb']]*len(self.inputL)] + [[[0]*self.h_layers[h+1]['neurNb']]*self.h_layers[h]['neurNb'] for h in range(len(self.h_layers)-1)] + [[[0]*len(self.outputL)]*self.h_layers[-1]['neurNb']]
		self.weights = [np.zeros((len(self.inputL), self.h_layers[0]['neurNb'] ))] + [np.zeros((self.h_layers[h]['neurNb'],self.h_layers[h+1]['neurNb'])) for h in range(len(self.h_layers)-1)] + [np.zeros((self.h_layers[-1]['neurNb'], len(self.outputL) ))]

		lenW = 0
		for a in self.weights:
			for b in a:
				lenW += len(b)
		self.nbOfW = lenW #Number of weights

		self.b_form = [[0]*h['neurNb'] for h in self.h_layers]+[len(self.outputL)*[0]]
		self.biases = [np.random.rand(h['neurNb']) for h in self.h_layers] + [np.random.rand(len(self.outputL))]
		self.activations = [np.zeros(len(self.inputL))]+ [np.zeros(h['neurNb']) for h in self.h_layers] + [np.zeros(len(self.outputL))]


	def saveNN(self, title = ''):
		title += '_'

		for l in self.h_layers:
			title += 'l' + str(l['neurNb'])

		title += '#' + str(self.nbOfW) + '#'

		title += 'EPOCHS(' + str(self.epochs) + ')'

		if self.self_train:
			title += "_SELF"

		if len(self.outputs) < 10:
			title += '_' + str(self.outputs).replace(' ', '').replace('[', '').replace(']', '').replace(',', '') + '_'

		title += '.nt'
		with open(title, 'wb') as output:  #Overwrites any existing file!
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
		print("Saved successfully")


	#Forward feed
	def recognition(self, image, isInput = False):
		#print("RECO")
		if self.image_input:
			image = image.reshape((784))/255
			prediction = image
		else:
			prediction = np.array([])
		intermediates = np.array([[0]*len(self.inputL)] + self.b_form)

		for nn in self.neural_inputs:
			for l in nn[1]:
				prediction = np.append(prediction, nn[0].recognition(image)[l])

		intermediates[0] = prediction
		for w in range(len(self.weights)):
			interm = np.matmul(prediction, self.weights[w])
			prediction = np.add(interm, self.biases[w])
			try:
				prediction = sigmoid(prediction)
			except:
				print(prediction)
				raise
			intermediates[w+1] = prediction

		return intermediates


	def training(self, data, l_rate = 0.2):
		startT = time()
		counterD = 0
		counterB = 0
		print("EPOCH:", self.epochs)
		errorT = np.array(self.b_form)
		thetaT = np.array(self.w_form)
		activationsT = np.array(self.activations)

		nbKnownOrNot = [0, 0]

		#Changing batch size (should be improved)
		batch_size = 2
		if self.epochs > 0:
			batch_size = 4
		if self.epochs > 2:
			batch_size = 8
		if self.epochs > 5:
			batch_size = 16
		if self.epochs > 6:
			batch_size = 32
		if self.epochs > 10:
			batch_size = 64
		if self.epochs > 18:
			batch_size = 128
		if self.epochs > 25:
			batch_size = 256

		print("batch_size:", batch_size)

		for d in data:
			d1 = d[1].reshape((784))/255
			error = np.array(self.b_form)
			activations = self.recognition(d[1])
			######
			if not self.self_train:
				target = np.array(len(self.outputs)*[0])
				if d[0] in self.outputs:
					target[self.outputs.index(d[0])] = 1
					nbKnownOrNot[0] += 1

				else:
					if nbKnownOrNot[1]*len(self.outputs)/((nbKnownOrNot[0]+1)) > 1.2:
						continue
					nbKnownOrNot[1] += 1
			######
			else:
				target = d1

			lenW = 0
			for a in self.weights:
				for b in a:
					lenW += len(b)

			#error backpropagation
			error = np.array(self.b_form)

			for l in range(1, len(error)+1):
				if l == 1:
					error[-l] = np.subtract(target,activations[-l])

				else:
					error[-l] = np.dot(self.weights[-l+1],np.array(error[-l+1]))
				error[-l] = error[-l]*deriv_sigm(activations[-l])

			for l in range(len(error)):
				if d[0] not in self.outputs:
					pass
				if not counterB:
					thetaT[l] = np.outer(activations[l].T,l_rate*error[l])
				else:
					thetaT[l] = np.add(thetaT[l], np.outer(activations[l].T,l_rate*error[l]))


			errorT = np.add(errorT, error)

			#Weights updates
			counterB += 1
			if counterB == batch_size:
				counterB = 0
				for l in range(0, len(error)):
					self.biases[l] = np.add(self.biases[l], l_rate*errorT[l])
					self.weights[l] = np.add(self.weights[l], thetaT[l])
				errorT = np.array(self.b_form)
				thetaT = np.array(self.w_form)
			counterD += 1
			if counterD%10000 == 0:
				pass
			if counterD%1000 == 0:
				pass
			#sys.stdout.write("\r{}/{}".format(str(counterD), str(60000)))
		self.epochs += 1

		totalT = time() - startT
		print("t:", totalT)


	def test(self, data):

		counterD = 0
		correctC = 0
		diffTotal = 0
		activation_threshold = 0.5
		for d in data:
			if not self.self_train:
				target = np.array(len(self.outputs)*[0])
				if d[0] in self.outputs:
					target[self.outputs.index(d[0])] = 1
				activations = self.recognition(d[1])

				pred = np.argmax(activations[-1])
				if len(self.outputs) < 10:
					if activations[-1][pred] < activation_threshold and d[0] not in self.outputs: #If no output activated more than activation_threshold considered as "other"
						correctC += 1

				if d[0] in self.outputs and self.outputs[pred] == d[0]:
					correctC += 1

			else:
				d1 = d[1].reshape((784))/255
				target = d1
				activations = self.recognition(d1)
				abs_v = np.vectorize(lambda x: abs(x))
				diff = np.sum(abs_v(activations[-1]-target))
				diffTotal += diff
			counterD += 1

			#sys.stdout.write("\r{}/{}".format(str(counterD), str('?')))
		if not self.self_train:
			print(correctC/counterD, "")
		else:
			print("Avg err per pixel: ", diffTotal/784/counterD )



def load_mnist(dataset = "training", path = "."):
	"""
	Python function for importing the MNIST data set.  It returns an iterator
	of 2-tuples with the first element being the label and the second element
	being a numpy.uint8 2D array of pixel data for the given image.
	"""

	print("dataset", dataset)
	if dataset == "training":
		fname_img = os.path.join(path, 'train-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
	elif dataset == "testing":
		fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
		fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
	else:
		raise ValueError #, "dataset must be 'testing' or 'training'"

	# Load everything in some numpy arrays
	with open(fname_lbl, 'rb') as flbl:
		magic, num = struct.unpack(">II", flbl.read(8))
		lbl = np.fromfile(flbl, dtype=np.int8)

	with open(fname_img, 'rb') as fimg:
		magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
		img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

	get_img = lambda idx: (lbl[idx], img[idx])

	# Create an iterator which returns each image in turn
	for i in range(len(lbl)):
		yield get_img(i)


def show(image):
	"""
	Render a given numpy.uint8 2D array of pixel data.
	"""
	from matplotlib import pyplot
	import matplotlib as mpl
	fig = pyplot.figure()
	ax = fig.add_subplot(1,1,1)
	imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
	imgplot.set_interpolation('nearest')
	ax.xaxis.set_ticks_position('top')
	ax.yaxis.set_ticks_position('left')
	pyplot.show()




if __name__ == '__main__':
	main()
