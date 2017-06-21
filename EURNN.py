from tensorflow.contrib.rnn.python.ops import *

from keras.engine.topology import Layer
from keras.layers import Recurrent
from keras.engine.topology import InputSpec

from EUNN import *

import tensorflow as tf
import random



def modReLU(z, b, comp):
	if comp:
		#z_plus_b_re = nn_ops.bias_add(z, b); 
		z_norm = math_ops.sqrt(math_ops.square(math_ops.real(z)) + math_ops.square(math_ops.imag(z))) + 0.000001
		step1 = z_norm + b*-1
		#nn_ops.bias_add(z_norm, -1)
		step2 = math_ops.complex(nn_ops.relu(step1), array_ops.zeros_like(z_norm))
		step3 = z/math_ops.complex(z_norm, array_ops.zeros_like(z_norm))
	else:
		z_norm = math_ops.abs(z) + 0.00001
		step1 = nn_ops.bias_add(z_norm, b)
		step2 = nn_ops.relu(step1)
		step3 = math_ops.sign(z)
		
	return math_ops.multiply(step3, step2)




class EURNNCell(Recurrent):


	def __init__(self, hidden_size, capacity=2, FFT=False, comp=False, activation=modReLU,**kwargs):
		print("My args: " , kwargs)
		super(EURNNCell,self).__init__(**kwargs)

		self._hidden_size = hidden_size
		self.units = hidden_size
		self.states = [None]
		self._activation = activation
		self._capacity = capacity
		self._FFT = FFT
		self._comp = comp
		self.state_spec = InputSpec(shape=(None,self.units))

		self.idNum = str(random.randint(1,1000))

		self.v1, self.v2, self.ind, self.diag, self._capacity = EUNN_param(hidden_size, capacity, FFT, comp,self.idNum)


	def build(self,input_shape):
		if isinstance(input_shape,list):
			input_shape = input_shape[0]
		self.input_dim = input_shape[2]
		with vs.variable_scope("eurnn_cell" + self.idNum):
			#self.U = self.add_weight(shape=(1,self.units),name="U",initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			#self.bias = self.add_weight(shape=(self.units,),name="bias",initializer=init_ops.constant_initializer())
			
			self.U = vs.get_variable("U"+self.idNum, [self.input_dim, self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
			self.bias = vs.get_variable("modReLUBias"+self.idNum, [self._hidden_size], initializer= init_ops.constant_initializer())
		self.built = True


	@property
	def state_size(self):
		return self._hidden_size

	@property
	def output_size(self):
		return self._hidden_size

	@property
	def capacity(self):
		return self._capacity

	# def __call__(self, inputs, state, scope=None):
	# 	with vs.variable_scope(scope or "eurnn_cell"):

	# 		Wh = EUNN_loop(state, self._capacity, self.v1, self.v2, self.ind, self.diag)
	# 		bias = None

	# 		if self._comp:
	# 			U_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
	# 			U_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
	# 			Ux_re = math_ops.matmul(inputs, U_re)
	# 			Ux_im = math_ops.matmul(inputs, U_im)
	# 			Ux = math_ops.complex(Ux_re, Ux_im)
	# 			bias_re = vs.get_variable("modReLUBias_re", [self._hidden_size], initializer= init_ops.random_uniform_initializer(-.01, .01))
	# 			bias_im = vs.get_variable("modReLUBias_im", [self._hidden_size], initializer= init_ops.random_uniform_initializer(-.01, .01))
	# 			bias = math_ops.complex(bias_re,bias_im)
	# 			#ones = tf.ones((self._hidden_size))
	# 			ones = vs.get_variable("mode_bias_two", [self._hidden_size], initializer= init_ops.constant_initializer(1))
	# 		else:
	# 			U = vs.get_variable("U", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
	# 			Ux = math_ops.matmul(inputs, U) 
	# 			bias = vs.get_variable("modReLUBias", [self._hidden_size], initializer= init_ops.constant_initializer())
	# 		output = self._activation(Ux+Wh+bias,ones,self._comp)
	# 		#output = self._activation((Ux + Wh), bias, self._comp)  

	# 	return output, output

	def step(self,inputs,states,scope=None):
		#state = states[0] #Hidden state
		print("State: " , states)
		#print inputs.get_shape()
		state = states[0]
		with vs.variable_scope("eurnn_cell"+self.idNum):

			Wh = EUNN_loop(state, self._capacity, self.v1, self.v2, self.ind, self.diag)
			bias = None

			if self._comp:
				U_re = vs.get_variable("U_re", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
				U_im = vs.get_variable("U_im", [inputs.get_shape()[-1], self._hidden_size], initializer=init_ops.random_uniform_initializer(-0.01, 0.01))
				Ux_re = math_ops.matmul(inputs, U_re)
				Ux_im = math_ops.matmul(inputs, U_im)
				Ux = math_ops.complex(Ux_re, Ux_im)
				bias_re = vs.get_variable("modReLUBias_re", [self._hidden_size], initializer= init_ops.random_uniform_initializer(-.01, .01))
				bias_im = vs.get_variable("modReLUBias_im", [self._hidden_size], initializer= init_ops.random_uniform_initializer(-.01, .01))
				bias = math_ops.complex(bias_re,bias_im)
				#ones = tf.ones((self._hidden_size))
				ones = vs.get_variable("mode_bias_two", [self._hidden_size], initializer= init_ops.constant_initializer(1))
			else:
				Ux = math_ops.matmul(inputs, self.U) 
			#output = self._activation(Ux+Wh+bias,ones,self._comp)
			output = self._activation((Ux + Wh), self.bias, self._comp)  

		return output, [output]

