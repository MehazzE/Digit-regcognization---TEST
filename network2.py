import numpy as np
import cv2

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
	
def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

class CrossEntropyCost(object):
	@staticmethod
	def fn(a,y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
		
	@staticmethod
	def delta(z,a,y):
		return (a-y)

class QuadraticCost(object):
	@staticmethod
	def fn(a,y):
		return 0.5*np.linalg.norm(a-y)**2
		
	@staticmethod
	def delta(z,a,y):
		return (a-y) * sigmoid_prime(z)
        

class Network(object):
	def __init__(self, sizes, cost=CrossEntropyCost):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.large_weight_initializer()
		self.cost = cost
		
	def default_weight_initializer(self):
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
		
	def large_weight_initializer(self):
		self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
	
	def load_weights(self, biases, weights):
		self.biases = biases
		self.weights = weights

	def feedfoward(self, a):
		for b,w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w,a) + b)
		return a
		
	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		activation = x
		activations = [x]
		zs = []
		
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		delta = (self.cost).delta(z[-1], activations[-1], y)
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		
		for l in range(2, self.num_layers):
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def calculate_loss(self, training_data):
		loss = 0
		for x,y in training_data:
			loss += self.cost.fn(self.feedfoward(x),y)
		return loss

	def update_mini_batch(self, mini_batch, eta, lmbda, n):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backprop(x,y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
		self.weights = [(1-eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w,nw in zip(self.weights, nabla_w)]
		