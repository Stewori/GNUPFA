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
Created on Sep 9, 2013

@author: Stefan Richthofer
'''
import mdp
import numpy as np
import PFACoreUtil as pfa
import PFANodeMDPRefImp as ref
from MergeLayer import Merger

class PFANode(mdp.Node):
	'''redefined _init_ Method with p, k as arguments'''
	def __init__(self, p = 2, k = 0, affine = True, input_dim = None, output_dim = None, dtype = None):
		super(PFANode, self).__init__(input_dim = input_dim, output_dim = output_dim, dtype = dtype)
		self.p = p
		self.k = k
		self.l = None
		self.affine = affine
		self.evThreshold = 0.0000000001
		self.sindex = -1
		self.maxindex = -1
		self.layerIndex = -1
		self.layerMax = -1

	'''Node is trainable'''
	def is_trainable(self):
		return True

	'''Saves relevant information of the data for further processing in stop_training.'''
	def _train(self, x):
		#print "PFA train "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		n = self.get_input_dim()
		x2 = x
		if not n is None:
			x2 = x.T[:n].T
		if self.l is None:
# 			self.data = x2
			self.startData = x2[:self.p+self.k]
			self.l = 1.0*len(x2)
			self.mean = x2.mean(0)
			#self.secondMoment = np.dot(x2.T, x2)/(1.0*len(x2))
			data_pk = x2[self.p+self.k:]
			self.corList = [np.dot(data_pk.T, data_pk)]
			for i in range(1, self.k+self.p+1):
				self.corList.append(np.dot(data_pk.T, x2[self.p+self.k-i:-i]))
		else:
			#Keeping always the averaged version instead of just accumulating here and finally dividing through
			#the whole length in _stop_training may be better for large data, since the magnitude of the matrix
			#entries is better kept in a sane range
# 			self.data = np.vstack([self.data, x2])
			self.mean = (self.l/(self.l+len(x2))) * self.mean  +  (len(x2)/(self.l+len(x2))) * x2.mean(0)
			#self.secondMoment = (self.l/(self.l+len(x2))) * self.secondMoment  +  (1.0/(self.l+len(x2))) * np.dot(x2.T, x2)
			#self.secondMoment = (self.l*self.secondMoment+np.dot(x2.T, x2))/(self.l+len(x2))
			self.corList[0] += np.dot(x2.T, x2)
			#self.corList[1] += np.dot(x2[1:].T, x2[:-1])+np.dot(x2[:1].T, self.endData[-1:])
			for i in range(1, self.k+self.p+1):
				self.corList[i] += np.dot(x2[i:].T, x2[:-i])+np.dot(x2[:i].T, self.endData[-i:])
			self.l += 1.0*len(x2)
		self.endData = x2[-self.p-self.k:]
	
	def _prepare_start_end_cor(self):
		start_cor = []
		end_cor = []
		for i in range(len(self.startData)):
			startLine = []
			endLine = []
			for j in range(len(self.startData)):
				if j < i:
					startLine.append(start_cor[j][i].T)
					endLine.append(end_cor[j][i].T)
				else:
					startLine.append(np.outer(self.startData[i], self.startData[j]))
					endLine.append(np.outer(self.endData[i], self.endData[j]))
			start_cor.append(startLine)
			end_cor.append(endLine)
		self.start_cor = start_cor
		self.end_cor = end_cor
	
	#This is seperate from _prepare_start_end_cor because merging would be done between these
	#Note that it requires that x_cor[i][j] is a reference to x_cor[j][i].T
	#This fact must be preserved during merging!
	def _start_end_cor_clear_mean(self):
		M = np.outer(self.mean, self.mean)
		for i in range(len(self.start_cor)):
			for j in range(i, len(self.start_cor[i])):
				self.start_cor[i][j] += -np.outer(self.startData[i], self.mean)-np.outer(self.mean, self.startData[j])+M
				self.end_cor[i][j] += -np.outer(self.endData[i], self.mean)-np.outer(self.mean, self.endData[j])+M
	
	def _stop_training(self):
		#print "PFA stop_training "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		if self.endData is None:
			raise mdp.TrainingException("train was never called")
		self._prepare_start_end_cor()
		self._start_end_cor_clear_mean()
		self.calc_PFA()
		
	def calc_PFA(self):
		#print "PFA calc_PFA "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		r = self.get_output_dim()
		if r is None:
			r = len(self.startData.T)
		
		#meanRef, SRef, zRef = pfa.calcSpheringParametersAndDataRefImp(self.data)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
		
# 		print "secondMomentTest:"
# 		print self.secondMoment*self.l
# 		print (self.corList[0]+np.dot(self.startData.T, self.startData))
# 		print "----------------"
# 		print "chunkTest:"
# 		print self.corList
		
		S = pfa.calcSpheringMatrixFromMeanAndSecondMoment(self.mean, (self.corList[0]+np.dot(self.startData.T, self.startData))/self.l, self.l, threshold = self.evThreshold)#, besselsCorrection = 0)
		self.S = S
		if S.shape[1] < r:
			r = S.shape[1]
			#this creates an inconsistent output_dim.
			#However, it does not matter, if one sets output_dim to an appropriate value
			#from the beginning.
			#self.output_dim = r
# 		print self.corList[0]
# 		print np.dot(self.startData.T, self.startData)
# 		corList00 = self.start_cor[0][0]
# 		for r2 in range(1, len(self.startData)):
# 			corList00 += self.start_cor[r2][r2]
# 		print corList00
# 		print "S"
# 		print S.shape
		
		meanList0 = [self.mean*self.l-self.startData.mean(0)*(self.p+self.k)]
		for i in range(1, self.k+self.p+1):
			meanList0.append(meanList0[-1]-self.endData[-i]+self.startData[-i])
		M = np.outer(self.mean*(self.l-self.p-self.k), self.mean)-np.outer(meanList0[0], self.mean)
		mns = np.outer(np.ones(self.p+self.k), self.mean)
		startZ = np.dot(self.startData-mns, S)
		endZ = np.dot(self.endData-mns, S)
		#startZ0 = self.startData-mns
		#endZ0 = self.endData-mns
		
		#print np.outer(startZ0[1], startZ0[2])
#		print np.outer(self.endData[0]-self.mean, self.endData[1]-self.mean)
#		print np.outer(self.endData[0], self.endData[1])-np.outer(self.mean, self.endData[1])-np.outer(self.endData[0], self.mean)+np.outer(self.mean, self.mean)
#		print np.outer(self.endData[0], self.endData[1])
		#print self.start_cor[1][2]
		
# 		zRef = np.dot(self.data-np.outer(np.ones(len(self.data)), self.mean), S)
# 		z_pk = zRef[self.p+self.k:]
# 		z_p = zRef[self.p:]
		corList = []#np.dot(z_pk.T, z_pk)]
		corList0 = []
		for i in range(0, self.k+self.p+1):
			corList.append(np.dot(S.T, np.dot(self.corList[i] - np.outer(self.mean, meanList0[i]) + M, S)))
			corList0.append(self.corList[i] - np.outer(self.mean, meanList0[i]) + M)

		zetaList = []
		lastLine = []
		for i in range(self.p):
			zetaLine = []
			for j in range(self.p):
				if j < i:
					zetaLine.append(zetaList[j][i].T)
				else:
					if i == 0:
						#zetaLine.append(corList[j]-np.outer(endZ[-1], endZ[-1-j])+np.outer(startZ[self.p+self.k-1], startZ[self.p+self.k-1-j]))
						#zetaLine.append(np.dot(S.T, np.dot(corList0[j]-np.outer(endZ0[-1], endZ0[-1-j])+np.outer(startZ0[self.p+self.k-1], startZ0[self.p+self.k-1-j]), S)))
						#zetaLine.append(np.dot(S.T, np.dot(corList0[j]-np.outer(endZ0[-1], endZ0[-1-j])+np.outer(startZ0[-1], startZ0[-1-j]), S)))
						zetaLine.append(np.dot(S.T, np.dot(corList0[j]-self.end_cor[-1][-1-j]+self.start_cor[-1][-1-j], S)))
					else:
						#zetaLine.append(lastLine[j-1]-np.outer(endZ[-1-i], endZ[-1-j])+np.outer(startZ[self.p+self.k-1-i], startZ[self.p+self.k-1-j]))
						#zetaLine.append(lastLine[j-1]+np.dot(S.T, np.dot(-np.outer(endZ0[-1-i], endZ0[-1-j])+np.outer(startZ0[self.p+self.k-1-i], startZ0[self.p+self.k-1-j]), S)))
						#zetaLine.append(lastLine[j-1]+np.dot(S.T, np.dot(-np.outer(endZ0[-1-i], endZ0[-1-j])+np.outer(startZ0[-1-i], startZ0[-1-j]), S)))
						zetaLine.append(lastLine[j-1]+np.dot(S.T, np.dot(-self.end_cor[-1-i][-1-j]+self.start_cor[-1-i][-1-j], S)))
			zetaList.append(zetaLine)
			lastLine = zetaLine
		for i in range(self.p):
			zetaList[i] = np.hstack(zetaList[i])
		
		corList_p = None
		zetaList_p = None
		lastLine_p = None
		if self.k > 0:
			corList_p = [corList[0]+np.dot(startZ[self.p:].T, startZ[self.p:])]#[np.dot(z_p.T, z_p)]
			#corList_p0 = [corList0[0]+np.dot(startZ0[self.p:].T, startZ0[self.p:])]
# 			R = np.array(self.start_cor[self.p][self.p])
# 			for r2 in range(self.p+1, len(self.startData)):
# 				R += self.start_cor[r2][r2]
# 			corList_p0 = [corList0[0]+R]
			corList_p0 = [corList0[0]+self.start_cor[self.p][self.p]]
			for r2 in range(self.p+1, len(self.startData)):
				corList_p0[0] += self.start_cor[r2][r2]
			
# 			corList_p00 = [np.dot(z_p.T, z_p)]
			for i in range(1, self.p+1):
# 				corList_p00.append(np.dot(z_p.T, zRef[self.p-i:-i]))
				corList_p.append(corList[i]+np.dot(startZ[self.p:].T, startZ[self.p-i:-i]))
				#corList_p0.append(corList0[i]+np.dot(startZ0[self.p:].T, startZ0[self.p-i:-i]))
# 				R = np.array(self.start_cor[self.p][self.p-i])
# 				for r2 in range(self.p+1, len(self.startData)):
# 					R += self.start_cor[r2][r2-i]
# 				corList_p0.append(corList0[i]+R)
				corList_p0.append(corList0[i]+self.start_cor[self.p][self.p-i])
				for r2 in range(self.p+1, len(self.startData)):
					corList_p0[i] += self.start_cor[r2][r2-i]
				
# 				for r in range(self.p+1, len(startZ0)):
# 					corList_p0[i] += self.start_cor[r][r-i]
# 				print "--------------"
# 				print np.dot(startZ0[self.p:].T, startZ0[self.p-i:-i])
# 				R = np.outer(startZ0[self.p], startZ0[self.p-i])
# 				for ir in range(self.p+1, len(startZ0)):
# 					R += np.outer(startZ0[ir], startZ0[ir-i])
# 				print R
			zetaList_p = []
			lastLine_p = []
			for i in range(self.p):
				zetaLine_p = []
				for j in range(self.p):
					if j < i:
						zetaLine_p.append(zetaList_p[j][i].T)
					else:
						if i == 0:
							#zetaLine_p.append(corList_p[j]-np.outer(endZ[-1], endZ[-1-j])+np.outer(startZ[self.p-1], startZ[self.p-1-j]))
							#zetaLine_p.append(np.dot(S.T, np.dot(corList_p0[j]-np.outer(endZ0[-1], endZ0[-1-j])+np.outer(startZ0[self.p-1], startZ0[self.p-1-j]), S)))
							zetaLine_p.append(np.dot(S.T, np.dot(corList_p0[j]-self.end_cor[-1][-1-j]+self.start_cor[self.p-1][self.p-1-j], S)))
						else:
							#zetaLine_p.append(lastLine_p[j-1]-np.outer(endZ[-1-i], endZ[-1-j])+np.outer(startZ[self.p-1-i], startZ[self.p-1-j]))
							#zetaLine_p.append(lastLine_p[j-1]+np.dot(S.T, np.dot(-np.outer(endZ0[-1-i], endZ0[-1-j])+np.outer(startZ0[self.p-1-i], startZ0[self.p-1-j]), S)))
							zetaLine_p.append(lastLine_p[j-1]+np.dot(S.T, np.dot(-self.end_cor[-1-i][-1-j]+self.start_cor[self.p-1-i][self.p-1-j], S)))
				zetaList_p.append(zetaLine_p)
				lastLine_p = zetaLine_p
			for i in range(self.p):
				zetaList_p[i] = np.hstack(zetaList_p[i])
		
		if not self.affine:
			zZ = np.hstack(corList[1:self.p+1])
			ZZ = np.vstack(zetaList)
			W = None
			if self.k == 0:
				ZZI = pfa.invertByProjectionRefImp(ZZ, self.evThreshold)
				W = np.dot(zZ, ZZI)
			else:
				zZ_p = np.hstack(corList_p[1:self.p+1])
				ZZ_p = np.vstack(zetaList_p)
				ZZ_pI = pfa.invertByProjectionRefImp(ZZ_p, self.evThreshold)
				W = np.dot(zZ_p, ZZ_pI)
			self.W0 = W
# 			WRef = pfa.calcRegressionCoeffRefImp(zRef, self.p, self.evThreshold)
# 			XRef = pfa.calcErrorCoeffConstLenRefImp(zRef, WRef, self.k)
			#X = pfa.calcErrorCoeffConstLenFromCorrelations2(W, zZ, ZZ, lastLine, corList, startZ, endZ, S, self.start_cor, self.end_cor, self.k)
			X = pfa.calcErrorCoeffConstLenFromCorrelations2(W, zZ, ZZ, lastLine, corList, S, self.start_cor, self.end_cor, self.k)
			#print "xxxx"
			#X = pfa.calcErrorCoeffConstLenFromCorrelations2(W, zZ, ZZ, lastLine, corList, S, self.start_cor, self.end_cor, self.k)
			
# 			print "WCompare:"
# 			print WRef
# 			print W
# 			print "XCompare:"
# 			print XRef
# 			print X
# 			print"------"
			self.X = X #needed only for debugging
			Ar = pfa.calcExtractionForErrorCovRefImp(X, r)
			#reduced = np.dot(self.data, self.Ar.T)
# 			reduced = np.dot(zRef, Ar)
# 			WRef = pfa.calcRegressionCoeffRefImp(reduced, self.p)
# 			print "Reduction compare"
			A_ = np.kron(np.identity(self.p), Ar)
			if (self.k == 0):
				zZ = np.dot(Ar.T, np.dot(zZ, A_))
				ZZ = np.dot(A_.T, np.dot(ZZ, A_))
				ZZI = pfa.invertByProjectionRefImp(ZZ, self.evThreshold)
				W = np.dot(zZ, ZZI)
			else:
				zZ_p = np.dot(Ar.T, np.dot(zZ_p, A_))
				ZZ_p = np.dot(A_.T, np.dot(ZZ_p, A_))
				ZZ_pI = pfa.invertByProjectionRefImp(ZZ_p, self.evThreshold)
				W = np.dot(zZ_p, ZZ_pI)
# 			print WRef
# 			print W
			self.W = W
# 			print "--------------------------"
			self.Ar = np.dot(S, Ar)
		else:
			meanList = [startZ.mean(0)*(-self.p-self.k)]
			for i in range(1, self.k+self.p+1):
				meanList.append(meanList[-1]-endZ[-i]+startZ[self.p+self.k-i])
			ml = np.hstack(meanList[1:self.p+1])
			zZ_c = np.vstack([np.hstack(corList[1:self.p+1]).T, meanList[0]]).T
			ZZ_c = np.vstack([np.vstack(zetaList), ml])
			ml1 = np.hstack([ml, [self.l-self.p-self.k]])
			ZZ_c = np.vstack([ZZ_c.T, ml1]).T
			
			W_c = None
			if self.k == 0:
				ZZ_cI = pfa.invertByProjectionRefImp(ZZ_c, self.evThreshold)
				W_c = np.dot(zZ_c, ZZ_cI)
			else:
				meanList_p = [startZ[:self.p].mean(0)*(-self.p)]
				for i in range(1, self.p+1):
					meanList_p.append(meanList_p[-1]-endZ[-i]+startZ[self.p-i])
				ml_p = np.hstack(meanList_p[1:self.p+1])
				zZ_pc = np.vstack([np.hstack(corList_p[1:self.p+1]).T, meanList_p[0]]).T
				ZZ_pc = np.vstack([np.vstack(zetaList_p), ml_p])
				ml1_p = np.hstack([ml_p, [self.l-self.p]])
				ZZ_pc = np.vstack([ZZ_pc.T, ml1_p]).T
				
				ZZ_pcI = pfa.invertByProjectionRefImp(ZZ_pc, self.evThreshold)
				W_c = np.dot(zZ_pc, ZZ_pcI)
			self.W0 = W_c
			#print self.start_cor
# 			WRef = pfa.calcRegressionCoeffAffineRefImp(zRef, self.p, self.evThreshold)
# 			XRef = pfa.calcErrorCoeffConstLenAffineRefImp(zRef, WRef, self.k)
			#X = pfa.calcErrorCoeffConstLenAffineFromCorrelations2(W_c, zZ_c, ZZ_c, lastLine, corList, meanList, self.l, startZ, endZ, self.k)
			X = pfa.calcErrorCoeffConstLenAffineFromCorrelations2(W_c, zZ_c, ZZ_c, lastLine, corList, meanList, self.l, S, self.start_cor, self.end_cor, self.k)
# 			print "WCompare (Affine):"
# 			print WRef
# 			print W_c
# 			print "XCompare (Affine):"
# 			print XRef
			#print X
# 			print"------"
			self.X = X #needed only for debugging
			Ar = pfa.calcExtractionForErrorCovRefImp(X, r)
# 			reduced = np.dot(zRef, Ar)
# 			WRef = pfa.calcRegressionCoeffAffineRefImp(reduced, self.p)
# 			print "Reduction compare (Affine)"
			A_ = pfa.kronAffine(Ar, self.p)
			if self.k == 0:
				zZ_c = np.dot(Ar.T, np.dot(zZ_c, A_))
				ZZ_c = np.dot(A_.T, np.dot(ZZ_c, A_))
				ZZ_cI = pfa.invertByProjectionRefImp(ZZ_c, self.evThreshold)
				W_c = np.dot(zZ_c, ZZ_cI)
			else:
				zZ_pc = np.dot(Ar.T, np.dot(zZ_pc, A_))
				ZZ_pc = np.dot(A_.T, np.dot(ZZ_pc, A_))
				ZZ_pcI = pfa.invertByProjectionRefImp(ZZ_pc, self.evThreshold)
				W_c = np.dot(zZ_pc, ZZ_pcI)
# 			print WRef
# 			print W_c
			self.W = W_c
# 			print "--------------------------"
			self.Ar = np.dot(S, Ar)
		#print "PFA_calc done "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)

	def _execute(self, x):
		#print "PFA execute "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		z0 = x-np.outer(np.ones(len(x)), self.mean)
		return np.dot(z0, self.Ar)

class PFANodeLayerAware(PFANode):
	'''redefined _init_ Method with p, k as arguments '''
	def __init__(self, train_length, p = 2, k = 0, affine = True, input_dim = None, output_dim = None, dtype = None):
		super(PFANodeLayerAware, self).__init__(p, k, affine, input_dim, output_dim, dtype)
		self.train_length = train_length
		self.field_index = 0
		self.current_train_index = 0
		self.l_acc = 0
	
	def _merge_init(self):
		self.mean_acc = np.array(self.mean)
		self.l_acc += self.l
		self.corList_acc = []
		for i in range(len(self.corList)):
			self.corList_acc.append(np.array(self.corList[i]))
		
		self.startData_acc = np.array(self.startData)
		self.endData_acc = np.array(self.endData)
		
		self.start_cor_acc = []
		self.end_cor_acc = []
		for i in range(len(self.startData)):
			start_line = []
			end_line = []
			for j in range(len(self.startData)):
				if j >= i:
					start_line.append(np.array(self.start_cor[i][j]))
					end_line.append(np.array(self.end_cor[i][j]))
				else:
					start_line.append(self.start_cor_acc[j][i].T)
					end_line.append(self.end_cor_acc[j][i].T)
			self.start_cor_acc.append(start_line)
			self.end_cor_acc.append(end_line)
	
	def _merge(self):
		#print "PFA inline-merge "+str(node.sindex)+" of "+str(node.maxindex)+"   layer "+str(node.layerIndex)+" of "+str(node.layerMax)
		self.mean_acc = (self.l_acc/(self.l_acc+self.l)) * self.mean_acc  +  (self.l/(self.l_acc+self.l)) * self.mean
		for i in range(len(self.corList_acc)):
			self.corList_acc[i] += self.corList[i]
			
		self.startData_acc += self.startData
		self.endData_acc += self.endData
			
		for i in range(len(self.startData_acc)):
			for j in range(len(self.startData_acc)):
				if j >= i:
					self.start_cor_acc[i][j] += self.start_cor[i][j]
					self.end_cor_acc[i][j] += self.end_cor[i][j]
						
		self.l_acc += self.l
	
	def _merge_scale(self):
		sc = 1.0*self.field_index
		self.startData_acc /= sc
		self.endData_acc /= sc
		
		self.l_acc /= sc
		for i in range(len(self.corList_acc)):
				self.corList_acc[i] /= sc
				
		for i in range(len(self.startData_acc)):
			for j in range(len(self.startData_acc)):
				if j >= i:
					self.start_cor_acc[i][j] /= sc
					self.end_cor_acc[i][j] /= sc
	
	def _merge_cor(self):
		if self.field_index == 1:
			self._merge_init()
		else:
			self._merge()
		self.l = None #causes _train to re-init everything but start_cor, end_cor
	
	def _train(self, x):
		self.current_train_index += len(x)
		super(PFANodeLayerAware, self)._train(x)
		if self.current_train_index == self.train_length:
			self.current_train_index = 0
			self.field_index += 1
			self._prepare_start_end_cor()
			self._merge_cor()
	
	def _insert_acc(self):
		self.mean = self.mean_acc
		self.l = self.l_acc
		self.corList = self.corList_acc
		self.startData = self.startData_acc
		self.endData = self.endData_acc
		self.start_cor = self.start_cor_acc
		self.end_cor = self.end_cor_acc
	
	def _stop_training(self):
		#print "PFA stop_training "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		if self.endData_acc is None:
			raise mdp.TrainingException("train was never called")
		
		self._merge_scale()
		self._insert_acc()
		self._start_end_cor_clear_mean()
		self.calc_PFA()
		

class PFAMerger(PFANode, Merger):
	def __init__(self, p = 2, k = 0, affine = True, input_dim = None, output_dim = None, dtype = None):
		#super(PFAMerger, self).__init__(p, k, affine, input_dim, output_dim, dtype)
		super(PFAMerger, self).__init__(p, k, affine, input_dim, output_dim, dtype)
		self.mergeCount = 0
		self.execCount = 0
		self.Ar = None
	
# 	def _execute(self, x):
# 		print "PFA merger execute"# ("+str(self.execCount)+") "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
# 		print "mean a1 "+str(self.mean)
# 		self.execCount += 1
# 		z0 = x-np.outer(np.ones(len(x)), self.mean)
# 		er = np.dot(z0, self.Ar)
# 		print "mean a2 "+str(self.mean)
# 		return er
	
	def _merge_init(self, node):
		if node.start_cor is None:
			node._prepare_start_end_cor()
			
		self.mean = np.array(node.mean)
		self.l = node.l
		self.corList = []
		for i in range(len(node.corList)):
			self.corList.append(np.array(node.corList[i]))
		
		self.startData = np.array(node.startData)
		self.endData = np.array(node.endData)
		
		self.start_cor = []
		self.end_cor = []
		for i in range(len(node.startData)):
			start_line = []
			end_line = []
			for j in range(len(node.startData)):
				if j >= i:
					start_line.append(np.array(node.start_cor[i][j]))
					end_line.append(np.array(node.end_cor[i][j]))
				else:
					start_line.append(self.start_cor[j][i].T)
					end_line.append(self.end_cor[j][i].T)
			self.start_cor.append(start_line)
			self.end_cor.append(end_line)
		self.mergeCount = 1
	
# 	def freeMem(self):
# 		#This hopefully allows the gc to free a lot of memory:
# 		self.start_cor = None
# 		self.end_cor = None
# 		self.mean = None
# 		self.corList = None
# 		self.startData = None
# 		self.endData = None
# 		self.l = None
# 		self.mergeCount += 1
	
	def _merge(self, node):
		#print "PFA merge "+str(node.sindex)+" of "+str(node.maxindex)+"   layer "+str(node.layerIndex)+" of "+str(node.layerMax)
		if self.mergeCount == 0:
			self._merge_init(node)
		else:
			if self.start_cor is None:
				#super(PFAMerger, self).__init__(p, k, affine, input_dim, output_dim, dtype)
				self._prepare_start_end_cor()
			if node.start_cor is None:
				node._prepare_start_end_cor()
			self.mean = (self.l/(self.l+node.l)) * self.mean  +  (node.l/(self.l+node.l)) * node.mean
			for i in range(len(self.corList)):
				self.corList[i] += node.corList[i]
			
			self.startData += node.startData
			self.endData += node.endData
			
			for i in range(len(self.startData)):
				for j in range(len(self.startData)):
					if j >= i:
						self.start_cor[i][j] += node.start_cor[i][j]
						self.end_cor[i][j] += node.end_cor[i][j]
						
			self.l += node.l
			self.mergeCount += 1
	
	def _merge_scale(self):
		#print "mergeCount: "+str(self.mergeCount)
		sc = 1.0*self.mergeCount
		self.startData /= sc
		self.endData /= sc
		
		self.l /= sc
		for i in range(len(self.corList)):
				self.corList[i] /= sc
				
		for i in range(len(self.startData)):
			for j in range(len(self.startData)):
				if j >= i:
					self.start_cor[i][j] /= sc
					self.end_cor[i][j] /= sc
	
	def _stop_training(self):
		#print "PFA merger stop training "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		self._prepare_start_end_cor()

	def _stop_merging(self):
		#print "PFA stop merging "+str(self.sindex)+" of "+str(self.maxindex)+"   layer "+str(self.layerIndex)+" of "+str(self.layerMax)
		self._merge_scale()
		self._start_end_cor_clear_mean()
		self.calc_PFA()
		
	def execute(self, x, *args, **kwargs):
		"""
		We skip _pre_execution_checks here, because train-flags
		might be in inconsistent states since the merger doesn't
		use train and stop training.
		Users of MergeLayer must know what they do.
		"""
		#self._pre_execution_checks(x)
		return self._execute(self._refcast(x), *args, **kwargs)

