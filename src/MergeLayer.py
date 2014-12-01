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
Created on Oct 16, 2013

@author: Stefan Richthofer
'''

from mdp.hinet.layer import Layer
from mdp import Node
from mdp.hinet import FlowNode

class Merger(Node):
	def is_trainable(self):
		"""Per default, a merger is not trainable, since it would not be trained except,
		it appears multiply in the network.
		"""
		return False
	
# 	def needs_stop_training_before_merge(self):
# 		return False
	
	### Methods to be implemented by the user

	# this are the methods the user has to overwrite

	def _merge(self, node):
		pass

	def _stop_merging(self):
		pass

	def freeMem(self):
		pass

	### User interface to the overwritten methods

	def merge(self, node):#, *args, **kwargs):
		"""Update the internal structures according to the input node `node`.

		`node` is an mdp-node that has been trained as part of a layer. The
		merger-subclass should exactly know about the node-type and the node's
		internal structure to retrieve the relevant data from it.

		By default, subclasses should overwrite `_merge` to implement their
		merging phase. The docstring of the `_train` method overwrites this
		docstring.
		"""

		self._merge(node)

	def stop_merging(self):#, *args, **kwargs):
		"""Stop the merging phase.

		By default, subclasses should overwrite `_stop_merging` to implement
		this functionality. The docstring of the `_stop_merging` method
		overwrites this docstring.
		"""
		self._stop_merging()#self, *args, **kwargs)
		#self._train_phase_started = True
		#self._training = False

class FlowMerger(Merger, FlowNode):
	"""
	Note that FlowMerger can be trainable though it is a merger.
	Edit: It should better not be trainable since this causes problems
	with train_phase and is_training.
	
	This is because the flow might contain trainable nodes as
	intermediate steps.
	You should use MergableFlowNodes instead of ordinary flows
	as merge sources. They ensure that the merger is used to
	execute the already trained/merged components.
	"""
	def __init__(self, merge_indices, flow, input_dim=None, output_dim=None, dtype=None):
		super(FlowMerger, self).__init__(flow, input_dim, output_dim, dtype)
		self.merge_indices = merge_indices
		self.current_merge = 0
		#self._train_phase = len(self._train_seq)
	
	def is_training(self):
		return False
	
# 	def needs_stop_training_before_merge(self):
# 		i = self.merge_indices[self.current_merge]
# 		if self.get_current_train_phase() == i:
# 			return self._flow.flow[i].needs_stop_training_before_merge()
# 		else:
# 			return True

	def _merge(self, node):
		i = self.merge_indices[self.current_merge]
		self._flow.flow[i].merge(node._flow.flow[i])
		node._flow.flow[i].freeMem()

	def _stop_merging(self):
		i = self.merge_indices[self.current_merge]
		self._flow.flow[i]._stop_merging()
		self.current_merge += 1
		#self.self._train_phase_started = True
		#self._training = False

# 	def force_stop_training(self, *args, **kwargs):
# 		"""Stop the training phase.
# 
# 		By default, subclasses should overwrite `_stop_training` to implement
# 		this functionality. The docstring of the `_stop_training` method
# 		overwrites this docstring.
# 		"""
# 		#if self.is_training() and self._train_phase_started == False:
# 		#    raise TrainingException("The node has not been trained.")
# 
# 		#if not self.is_training():
# 		#	err_str = "The training phase has already finished."
# 		#	raise TrainingFinishedException(err_str)
# 
# 		# close the current phase.
# 		self._train_seq[self._train_phase][1](*args, **kwargs)
# 		self._train_phase += 1
# 		self._train_phase_started = False
# 		# check if we have some training phase left
# 		if self.get_remaining_train_phase() == 0:
# 			self._training = False

class MergableFlowNode(FlowNode):
	
	#def __init__(self, flow, input_dim=None, output_dim=None, dtype=None):
	def __init__(self, flow, merger, input_dim, output_dim, dtype):
		super(MergableFlowNode, self).__init__(flow, input_dim, output_dim, dtype)
		self.merger = merger #None
		#self._train_seq_cache = None
	
	#def set_merger(self, merger):
	#	self.merger = merger
	
# 	def _get_train_seq(self):
# 		if self._train_seq_cache is None:
# 			self._train_seq_cache = self._build_train_seq()
# 		return self._train_seq_cache
# 	
# 	def _build_train_seq(self):
	def _get_train_seq(self):
		"""
		Return a training sequence containing all training phases.
		In contrast to the original FlowNode, MergableFlowNode uses
		a given merger (must be provided by a call to set_merger)
		to process the data through the already trained part.
		"""
		def get_train_function(_i_node, _node):
			# This internal function is needed to channel the data through
			# the nodes in front of the current nodes.
			# using nested scopes here instead of default args, see pep-0227
			def _train(x, *args, **kwargs):
				if i_node > 0:
					#_node.train(self._flow.execute(x, nodenr=_i_node-1), *args, **kwargs)
					#print "delegate exec in train to merger "
					#print str(self.merger._flow.flow[0].Ar)
					_node.train(self.merger._flow.execute(x, nodenr=_i_node-1), *args, **kwargs)
					#_node.train(self.merger.execute(x), *args, **kwargs)
				else:
					_node.train(x, *args, **kwargs)
					#self.merger._train_phase_started = True
			return _train
		
		train_seq = []
		for i_node, node in enumerate(self._flow):
			if node.is_trainable():
				remaining_len = (len(node._get_train_seq())
								 - self._pretrained_phase[i_node])
				train_seq += ([(get_train_function(i_node, node),
								node.stop_training)] * remaining_len)

		# try fix the dimension of the internal nodes and the FlowNode
		# after the last node has been trained
		def _get_stop_training_wrapper(self, node, func):
			def _stop_training_wrapper(*args, **kwargs):
				func(*args, **kwargs)
				self._fix_nodes_dimensions()
			return _stop_training_wrapper
		
		if train_seq:
			train_seq[-1] = (train_seq[-1][0],
							 _get_stop_training_wrapper(self, self._flow[-1], train_seq[-1][1]))
		return train_seq
	
	
	def train(self, x, *args, **kwargs):
		print "MergableFlowNode train "+str(self._train_phase)
		super(MergableFlowNode, self).train(x, *args, **kwargs)
	
	
	def skip_training(self):
		"""
		Close the current phase without actually performing it.
		This is useful if we know that it was already performed for the
		current inner node from a different reference.
		Only thing left to do is inform the surrounding flow.
		This method does it.
		""" 
		#node._train_seq[self._train_phase][1](*args, **kwargs)
		self._train_phase += 1
		self._train_phase_started = False
		# check if we have some training phase left
		if self.get_remaining_train_phase() == 0:
			self._training = False

class MergeLayer(Layer):
	"""Layer with an additional merging-phase that merges all nodes in the layer
	after training. Merging is done by a given merger, which is itself a node.
	After merging, the merger will be used for execution in a CloneLayer-like
	fashion.

	The idea behind MergeLayer is a hybrid of ordinary layer and CloneLayer.
	The goal in this design is to use separate nodes in the train-phase,
	while using only a single node for execution. The difference to CloneLayer
	is that in MergeLayer, a different algorithm can be used for combining
	horizontally parallel data chunks than for combining time-sequent data
	chunks. The latter ones are combined by the nodes in the usual train-phase.
	In Contrast to CloneLayer, MergeLayer allows to control how horizontal merging
	of the data works. While CloneLayer would push this data into the very same
	train method like the time-sequent chunks, MergeLayer uses a merger to combine
	horizontal data.
	"""

	def __init__(self, merger, nodes, call_stop_training = False, dtype=None):
		"""Setup the layer with the given list of nodes.

		Keyword arguments:
		merger -- Merger to be used.
		nodes -- List of the nodes to be used.
		"""
		super(MergeLayer, self).__init__(nodes, dtype=dtype)
		self.merger = merger
		self.call_stop_training = call_stop_training

	def _stop_training(self, *args, **kwargs):
		"""Stop training of the internal node."""
		if self.call_stop_training:
			super(MergeLayer, self)._stop_training()
		for node in self.nodes:
			self.merger.merge(node)
		self.merger.stop_merging()
		self.trained_nodes = self.nodes
		self.nodes = (self.merger,) * len(self.trained_nodes)
		if self.output_dim is None:
			self.output_dim = self._get_output_dim_from_nodes()
			
class FlowMergeLayer(Layer):
	"""Like MergeLayer, but more aware of multiple training phases.
	The inner nodes must be MergableFlows.
	Given that these need multiple training phases, AdvancedMergeLayer
	can treat some phases like MergeLayer, others like CloneLayer or ordinary Layer.
	If the flows contain non-merge steps with a reused
	Node-reference, this reference should not have stop_training called
	several times.
	
	Warning: Nesting of FlowMergers is not supported!
	
	If the index of a training phase appears in merge_indices, it calls:
	
	super(FlowMergeLayer).stop_training
	for node in self.nodes:
		self.merger.merge(node)
		self.merger.stop_merging()
		
	If the index appears in no list, it calls:
	
	super(FlowMergeLayer).stop_training()
	
	If the index appears in clone_indices, it calls:
	
	merger.stop_training()
	for node in self.nodes:
		#close training phase without calling stop_training:
		node.skip_training()
	
	It expects merger to contain the same node reference as the usual inner
	nodes. (CloneLayer-like fashion but taking merger as the reference handle).
	Expecting the inner nodes of the layer to be different flows containing the
	same node reference, the flow's training phase is closed without actually
	performing it (it was already performed by merger.stop_training).
	
	The lists are expected to contain the indices in strictly ascending order.
	
	Note that unlike merge layer, the call of layer-wide stop_training can't be
	avoided, since the training phase index must increase. So AdvancedMergeLayer
	has no parameter call_stop_training.
	
	todo: Maybe find better solution for merge/train sequence
	Use mode list instead of index list: [merge, clone, train, train, clone, merge,...]
	"""

	def __init__(self, merger, nodes, merge_indices, clone_indices, dtype=None):
		"""Setup the layer with the given list of nodes.

		Keyword arguments:
		merger -- Merger to be used. Must be a FlowMerger.
		nodes -- List of the nodes to be used. These must be MergableFlows.
		"""
		#for node in nodes:
			#node.set_merger(merger)
		super(FlowMergeLayer, self).__init__(nodes, dtype)
		self.merger = merger
		self.merge_indices = merge_indices
		self.clone_indices = clone_indices
		#offsets:
		self.current_merge = 0
		self.current_clone = 0
		self.current_train = 0

	def _stop_training(self, *args, **kwargs):
		"""Stop training of the internal node."""
		#print "FlowMerger stop training"
		if self.current_merge < len(self.merge_indices) and self.current_train == self.merge_indices[self.current_merge]:
			#print "FlowMerger.super stop training..."
			#super(FlowMergeLayer, self).stop_training()
			for node in self.nodes:
				node.stop_training()
			#print "FlowMerger.super stop training done"
			#self._train_phase
			for node in self.nodes:
				self.merger.merge(node)
			self.merger.stop_merging()
			#print "stop merging done...."
			self.current_merge += 1
		elif self.current_clone < len(self.clone_indices) and self.current_train == self.clone_indices[self.current_clone]:
			#self.merger.force_stop_training()
			#self.merger.skip_training()
			self.nodes[0].stop_training()
			for i in range(1, len(self.nodes)):
				self.nodes[i].skip_training()
			self.current_clone += 1
		else:
			#I currently see no usecase for this. Maybe remove it...
			super(FlowMergeLayer, self).stop_training()
		self.current_train += 1
		#print "Close Layer training? "+str(self.get_remaining_train_phase())
		if self.get_remaining_train_phase() == 1: #This already indicates the last phase, because  stop_training decrements it after the call to _stop_training
			self.trained_nodes = self.nodes
			self.nodes = (self.merger,) * len(self.trained_nodes)
			#print "Inserted merger as nodes"
			if self.output_dim is None:
				self.output_dim = self._get_output_dim_from_nodes()
