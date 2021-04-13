import math
from random import random


class NN:
	def __init__(self):
		self.wi1 = random()
		self.wi2 = random()
		self.wo = random()
		self.bh = random()
		self.bo = random()
		self.z1 = random()
		self.z2 = random()
		self.a1 = random()
		self.a2 = random()

	def feed_forward(self, i1, i2):
		self.z1 = self.wi1 * i1 + self.wi2 * i2 + self.bh
		self.a1 = sigmoid(self.z1)
		self.z2 = self.wo * self.a1 + self.bo
		self.a2 = sigmoid(self.z2)
		return self.a2

	def calc_error(self, i1, i2, d):
	 return (self.feed_forward(i1, i2) - d) ** 2.0

	def back_prop(self, i1, i2, d):
		error = self.calc_error(i1, i2, d)
		d_error = 2.0 * (self.a2 - d)
		d_z2 = sigmoid_der(self.z2)
		d_Cwo = d_error * d_z2 * self.a1
		d_Cbo = d_error * d_z2
		d_Ca1 = d_error * d_z2 * self.wo
		d_z1 = sigmoid_der(self.z1)
		d_Cwi1 = d_error * d_z2 * self.wo * d_z1 * i1
		d_Cwi2 = d_error * d_z2 * self.wo * d_z1 * i2
		d_Cbh = d_error * d_z2 * self.wo * d_z1

		self.wo += -d_Cwo
		self.bo += -d_Cbo
		self.wi1 += -d_Cwi1
		self.wi2 += -d_Cwi2
		self.bh += -d_Cbh


def sigmoid(x):
	return 1.0 / (1.0 + math.e ** (-x))

def sigmoid_der(x):
	return sigmoid(x) * (1.0 - sigmoid(x))


train_set = [
	(0.0, 0.0, 0.0),
	(0.0, 1.0, 0.0),
	(1.0, 0.0, 0.0),
	(1.0, 1.0, 1.0)
]


if __name__ == '__main__':
	brain = NN()
	for i in range(1000):
		t_params = train_set[i%4]
		brain.back_prop(t_params[0], t_params[1], t_params[2])
	for t in train_set:
		print("SET: {} -> {}".format(t, brain.feed_forward(t[0], t[1])))
		print("ERROR: {}".format(brain.calc_error(t[0], t[1], t[2])))
