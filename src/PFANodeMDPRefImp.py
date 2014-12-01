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
Created on 13.08.2013

@author: Stefan Richthofer
'''
import mdp
import numpy as np
import PFACoreUtil as pfa

class PFANode(mdp.Node):
	'''redifined _init_ Method with p, k as arguments '''
	def __init__(self, p = 2, k = 0, affine = True, input_dim = None, output_dim = None, dtype = None):
		super(PFANode, self).__init__(input_dim = input_dim, output_dim = output_dim, dtype = dtype)
		self.p = p
		self.k = k
		self.data = None
		self.affine = affine

	''' Node is trainable '''
	def is_trainable(self):
		return True

	'''In this reference implementation, it simply collects the data to process it in _stop_training.'''
	def _train(self, x):
		n = self.get_input_dim()
		x2 = x
		if not n is None:
			x2 = x.T[:n].T
		if self.data is None:
			self.data = x2
		else:
			self.data = np.vstack([self.data, x2])

	def _stop_training(self):
		if self.data is None:
			raise TrainingException("train was never called")
		r = self.get_output_dim()
		if r is None:
			r = len(self.data.T)
		meanRef, SRef, zRef = pfa.calcSpheringParametersAndDataRefImp(self.data)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
		self.mean = meanRef
		if not self.affine:
			WRef = pfa.calcRegressionCoeffRefImp(zRef, self.p)
			XRef = pfa.calcErrorCoeffConstLenRefImp(zRef, WRef, self.k)
			self.Ar = pfa.calcExtractionForErrorCovRefImp(XRef, r)
			reduced = np.dot(self.data, self.Ar.T)
			self.W = pfa.calcRegressionCoeffRefImp(reduced, self.p)
		else:
			WRef = pfa.calcRegressionCoeffAffineRefImp(zRef, self.p)
			XRef = pfa.calcErrorCoeffConstLenAffineRefImp(zRef, WRef, self.k)
			Ar = pfa.calcExtractionForErrorCovRefImp(XRef, r)
			reduced = np.dot(zRef, Ar)
			self.W = pfa.calcRegressionCoeffAffineRefImp(reduced, self.p)
			self.Ar = np.dot(SRef, Ar)

	def _execute(self, x):
		z0 = x-np.outer(np.ones(len(x)), self.mean)
		return np.dot(z0, self.Ar.T)

