#
#  
#  Copyright of GNUPFA:
#  Copyright (c) 2013, 2014  Institut fuer Neuroinformatik,
#  Ruhr-Universitaet Bochum, Germany.  All rights reserved.
#
#
#  This file is part of GNUPFA.
#
#  GNUPFA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  GNUPFA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with GNUPFA.  If not, see <http://www.gnu.org/licenses/>.
#
#
#  Linking this library statically or dynamically with other modules is
#  making a combined work based on this library.  Thus, the terms and
#  conditions of the GNU General Public License cover the whole
#  combination.
#


'''
Created on Dec 21, 2013

@author: Stefan Richthofer
'''
from mdp import Node
from mdp import TrainingException
from mdp.hinet import FlowNode
from mdp.hinet.layer import CloneLayer
import numpy as np

class CacheNode(Node):

	def __init__(self, filename, cacheSize = -1, input_dim=None, output_dim=None, dtype=None):
		super(CacheNode, self).__init__(input_dim, output_dim, dtype)
		self.cacheName = filename
		self.cacheSize = cacheSize
		self.cache = None #np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
		self.cachePos = 0
		self.defaultOutputLength = 0
		self.cacheLength = 0

	def reshape(self, shape):
		if self.cache is None:
			return
		s = self.cache.dtype.itemsize
		for n in shape:
			s *= n
		self.cache._mmap.resize(s)
		del self.cache
		self.cache = np.memmap(self.cacheName, dtype=self.dtype, mode='readwrite', shape=shape)

	#Used to fill the cache
	def _train(self, x):
# 		print self.dtype
		if len(x) > self.defaultOutputLength:
			self.defaultOutputLength = len(x)
		self.cacheLength += len(x)
		if self.cache is None:
			if self.cacheSize == -1:
				#self.cache = np.memmap(self.cacheName, dtype='float32', mode='w+', shape = x.shape)
				self.cache = np.memmap(self.cacheName, dtype=self.dtype, mode='w+', shape = x.shape)
			else:
				#self.cache = np.memmap(self.cacheName, dtype='float32', mode='w+', shape = (self.cacheSize, len(x[0])))
				self.cache = np.memmap(self.cacheName, dtype=self.dtype, mode='w+', shape = (self.cacheSize, len(x[0])))
		elif self.cacheSize == -1:
			self.reshape((self.cache.shape[0]+len(x), len(x[0])))
# 			print x[0][0].dtype.itemsize
# 			print self.cache._mmap.size()
# 			#self.cache._mmap.resize( (self.cache.shape[0]+len(x), len(x[0])) )
# 			print self.cache.shape
# 			newShape = (self.cache.shape[0]+len(x), len(x[0]))
# 			memmap_resize( newShape, self.cache )
# 			del self.cache
# 			self.cache = np.memmap(self.cacheName, dtype=self.dtype, mode='w+', shape = newShape)
# 			print "new size: "+str(self.cache._mmap.size())
# 			print self.cache.reshape(newShape)
		self.cache[self.cachePos:self.cachePos+len(x)] = x
# 		print self.cache._mmap.size()
# 		print self.cache[0][0]
# 		print self.cache[0][0].dtype.itemsize
# 		print "---"
		self.cachePos += len(x)

	def _stop_training(self):
		self.cachePos = 0

	def read(self, off, read_len = -1):
		if self.cache is None:
			raise TrainingException("CacheNode was not filled (i.e. trained).")
		if off >= self.cacheLength:
			return None
		l = read_len
		if l == -1:
			l = self.defaultOutputLength
		if off+l > self.cacheLength:
			l = self.cacheLength-off
		return self.cache[off:off+l]

	def _execute(self, x):
		er = self.read(self.cachePos, len(x))
		self.cachePos += len(x)
		return er

class ReorderingCacheNode(CacheNode):

	def __init__(self, fieldsize, filename, cacheSize = -1, input_dim=None, output_dim=None, dtype=None):
		super(ReorderingCacheNode, self).__init__(filename, cacheSize, input_dim, fieldsize, dtype)
		self.fieldsize = fieldsize

	def _train(self, x):
		self.cacheLength += len(x)*len(x[0])/self.fieldsize
		if len(x)*len(x[0])/self.fieldsize > self.defaultOutputLength:
			self.defaultOutputLength = len(x)*len(x[0])/self.fieldsize
		if self.cache is None:
			if self.cacheSize == -1:
				self.cache = np.memmap(self.cacheName, dtype=self.dtype, mode='w+', shape = x.shape)
			else:
				self.cache = np.memmap(self.cacheName, dtype=self.dtype, mode='w+', shape = (self.cacheSize, len(x[0])))
		elif self.cacheSize == -1:
			self.reshape( (self.cache.shape[0]+len(x), len(x[0])) )
		self.cache[self.cachePos:self.cachePos+len(x)] = x
		self.cachePos += len(x)

	def read(self, off, read_len = -1):
		if self.cache is None:
			raise TrainingException("CacheNode was not filled (i.e. trained).")
		if off >= self.cacheLength:
			return None
		l = read_len
		if l == -1:
			l = self.defaultOutputLength
		if off+l > self.cacheLength:
			l = self.cacheLength-off
		off1 = off % len(self.cache)
		if off1+l > len(self.cache):
			l = len(self.cache)-off1
		off2 = (off / len(self.cache)) * self.fieldsize
		return self.cache[off1:off1+l, off2:off2+self.fieldsize]

	def read_in_order(self, off, read_len = -1):
		return super(ReorderingCacheNode, self).read(off, read_len)

	def _execute(self, x):
		er = self.read(self.cachePos, len(x)*len(x[0])/self.fieldsize)
		self.cachePos += len(x)
		return er

def contains(array, value):
	for i in range(len(array)):
		if array[i] == value:
			return True
	return False

def buildCaches(cacheIndices, length, basename, cacheSize = -1):
	caches = []
	for i in range(length):
		if (cacheIndices is None) or contains(cacheIndices, i):
			caches.append(CacheNode(basename+"_"+str(i), cacheSize))
		else:
			caches.append(None)
	return caches

class CachingFlowNode(FlowNode):

	def __init__(self, flow, caches = [], cacheSize = -1, input_dim=None, output_dim=None, dtype=None):
		super(CachingFlowNode, self).__init__(flow, input_dim, output_dim, dtype)
		self._train_seq_cache = None
		self.caches = caches

	def _check_nodes_consistency(self, flow = None):
		"""Check the dimension consistency of a list of nodes."""
		if flow is None:
			flow = self._flow.flow
		for i in range(1, len(flow)):
			out = flow[i-1].output_dim
			if not self.caches[i] is None:
				out = self.caches[i].output_dim
			inp = flow[i].input_dim
			self._flow._check_dimension_consistency(out, inp)

	def _fix_nodes_dimensions(self):
		"""Try to fix the dimensions of the internal nodes."""
		if len(self._flow) > 1:
			prev_node = self._flow[0]
			for node in self._flow[1:]:
				if node.input_dim is None:
					node.input_dim = prev_node.output_dim
				prev_node = node
			self._check_nodes_consistency()
		if self._flow[-1].output_dim is not None:
			# additional checks are performed here
			self.output_dim = self._flow[-1].output_dim

	def find_first_cache(self):#, _i_node):
		#print "n: "+str(_i_node)
		for i in range(len(self.caches)):#_i_node+1):
			if not (self.caches[i] is None):
				return i
		return -1

	def _get_train_seq(self):
		"""Return a training sequence containing all training phases."""
		def get_train_function(_i_node, _node):
			# This internal function is needed to channel the data through
			# the nodes in front of the current nodes.
			# using nested scopes here instead of default args, see pep-0227
			#cachePos = self.find_nearest_cache(_i_node)
			def _train(x, *args, **kwargs):
				if i_node > 0:
					_node.train(self._flow.execute(x, nodenr=_i_node-1), *args, **kwargs)
				#	if cachePos == _i_node:
				#		_node.train(self.caches[cachePos].execute(x), *args, **kwargs)
				#	elif cachePos == -1:
				#		_node.train(self._flow.execute(x, nodenr=_i_node-1), *args, **kwargs)
				#	else:
				#		y = self.caches[cachePos].execute(x)
				#		for i in range(cachePos, _i_node-1):
				#			y = self._flow.flow[i].execute(y)
				#		_node.train(y, *args, **kwargs)
				else:
					_node.train(x, *args, **kwargs)
			return _train
		
		train_seq = []
		startCache = self.find_first_cache()
		if startCache == -1:
			startCache = len(self._flow.flow)
		
		#for i_node, node in enumerate(self._flow):
		for i_node in range(startCache):
			node = self._flow[i_node]
			if node.is_trainable():
				remaining_len = (len(node._get_train_seq()) - self._pretrained_phase[i_node])
				train_seq += ([(get_train_function(i_node, node), node.stop_training)] * remaining_len)
		
		if not self.caches[startCache] is None:
			def _train_tail(x, *args, **kwargs):
				if startCache > 0:
					self.caches[startCache].train(self._flow.execute(x, nodenr=startCache-1), *args, **kwargs)
				else:
					self.caches[startCache].train(x)
	
			def _stop_train_tail(*args, **kwargs):
				startCache = self.find_first_cache()
				if startCache == -1:
					startCache = len(self._flow.flow)
				currentTrainNode = startCache
				while currentTrainNode < len(self._flow.flow):
					#trainNode = self._flow.flow[currentTrainNode]
					if self._flow.flow[currentTrainNode].is_trainable():
						remaining_len = (len(self._flow.flow[currentTrainNode]._get_train_seq()) - self._pretrained_phase[i_node])
						for i in range(remaining_len):
							readOff = 0
							trainData = self.caches[startCache].read(readOff)
							while not trainData is None:
								for j in range(startCache, currentTrainNode):
									trainData = self._flow.flow[j].execute(trainData)
								self._flow.flow[currentTrainNode].train(trainData, *args, **kwargs)
								readOff += len(trainData)
								trainData = self.caches[startCache].read(readOff)
							self._flow.flow[currentTrainNode].stop_training(*args, **kwargs)
					currentTrainNode += 1
					if len(self.caches) < currentTrainNode and not self.caches[currentTrainNode] is None:
						readOff = 0
						trainData = self.caches[startCache].read(readOff)
						while not trainData is None:
							for j in range(startCache, currentTrainNode):
								trainData = self._flow.flow[j].execute(trainData)
							self.caches[currentTrainNode].train(trainData, *args, **kwargs)
							readOff += len(trainData)
							trainData = self.caches[startCache].read(readOff)
						self.caches[currentTrainNode].stop_training(*args, **kwargs)
						startCache = currentTrainNode
	
				self._fix_nodes_dimensions()
	
			train_seq.append((_train_tail, _stop_train_tail))
		
		else:

			# try fix the dimension of the internal nodes and the FlowNode
			# after the last node has been trained
			def _get_stop_training_wrapper(self, node, func):
				def _stop_training_wrapper(*args, **kwargs):
					func(*args, **kwargs)
					self._fix_nodes_dimensions()
				return _stop_training_wrapper
		
			if train_seq:
				train_seq[-1] = (train_seq[-1][0], _get_stop_training_wrapper(self, self._flow[-1], train_seq[-1][1]))
				
		return train_seq
		
class CacheCloneLayer(CloneLayer):#, CacheNode):
	#multi inheritance fails because mro would delgate __init__ call in Layer to CacheNode instead of Node.
	
	"""A CloneLayer variant that caches all training data to disc and
	then trains the backing node with reordered data. The data is reordered
	such that each field is completely trained (i.e. all chunks, if multiple
	chunks are used) before the next field starts.
	"""

	def __init__(self, cacheName, node, n_nodes=1, dtype=None):
		"""Setup the layer with the given list of nodes.

		Keyword arguments:
		node -- Node to be cloned.
		n_nodes -- Number of repetitions/clones of the given node.
		"""
		
		super(CacheCloneLayer, self).__init__(node=node, n_nodes=n_nodes, dtype=dtype)
		self.node = node  # attribute for convenience
		self.cache = ReorderingCacheNode(node.input_dim, cacheName)
		self.cacheLength = 0

	def _get_train_seq(self):
		return [(self._train, self._stop_training)]

	def read(self, off, read_len = -1):
		return self.cache.read_in_order(off, read_len)

# 	def _get_train_seq(self):
# 		"""Return the train sequence.
# 
# 		The length is set by the node with maximum length.
# 		"""
# 		max_train_length = 0
# 		for node in self.nodes:
# 			node_length = len(node._get_train_seq())
# 			if node_length > max_train_length:
# 				max_train_length = node_length
# 		return ([[self._train, self._stop_training]] * max_train_length)

	def _train(self, x, *args, **kwargs):
		self.cacheLength += len(x)
		self.cache.train(x)
# 		"""Perform single training step by training the internal nodes."""
# 		start_index = 0
# 		stop_index = 0
# 		for node in self.nodes:
# 			start_index = stop_index
# 			stop_index += node.input_dim
# 			if node.is_training():
# 				node.train(x[:, start_index : stop_index], *args, **kwargs)

# 	def _stop_training(self, *args, **kwargs):
# 		"""Stop training of the internal nodes."""
# 		for node in self.nodes:
# 			if node.is_training():
# 				node.stop_training(*args, **kwargs)
# 		if self.output_dim is None:
# 			self.output_dim = self._get_output_dim_from_nodes()

	def _stop_training(self, *args, **kwargs):
		"""Stop training of the internal node."""
		self.cache.stop_training()
		phases = self.node.get_remaining_train_phase()
		for i in range(phases):
			pos = 0
			while pos < self.cache.cacheLength:
				x = self.cache.read(pos)
				pos += len(x)
				self.node.train(x)
			self.node.stop_training(*args, **kwargs)
		#if self.node.is_training():
		#	self.node.stop_training(*args, **kwargs)
		if self.output_dim is None:
			self.output_dim = self._get_output_dim_from_nodes()
