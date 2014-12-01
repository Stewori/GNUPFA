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


import numpy as np
from numpy import linalg as LA

def NonSquareIdentity(shape, rowOffset = 0, columnOffset = 0):
	er = np.zeros(shape)
	l = shape[0]
	if shape[1] < l:
		l = shape[1]
	for i in range(l):
		er[rowOffset+i][columnOffset+i] = 1.0
	return er

#W is expected to be a broad matrix, transform would be multiplied from left to it, inverse transform (kron) from right
#Transform is expected to be orthogonal (or orthogonal subbase)
def transformPredictor(W, transform):
	X_ = np.kron(np.identity(len(W.T)/len(W)), transform.T)
	if len(W.T)%len(W) == 0: #Non-affine
		return np.dot(transform, np.dot(W, X_))
	else: #affine
		return np.hstack([np.dot(transform, np.dot(W.T[:-1].T, X_)), np.dot(transform, W.T[-1:].T)])
	
def kronAffine(A, p):
	X_ = np.kron(np.identity(p), A)
	z = np.zeros([len(X_), 1])
	X_ = np.hstack([X_, z])
	z = np.zeros([1, len(X_.T)])
	z[0][-1] = 1.0
	return np.vstack([X_, z])

#Note that secondMoment is expected without besselsCorrection, i.e. secondMomentSum/length rather than secondMomentSum/(length-1)
def calcSpheringMatrixFromMeanAndSecondMoment(mean, secondMoment, length, threshold = 0.00000000001, besselsCorrection = 0):
	"""
	Returns sphered version of the given data and the sphering parameters as well.
	The data is expected in the format (time, dims). The sphering matrix may reduce
	the dimensionality of the data by deleting dimensions that have (near) zero
	variance. The returned sphering matrix has the shape (high dim, low dim).
	The sphering is done with respect to a covariance-matrix including Bessel's correction,
	if the besselsCorrection parameter is True (default is False).
	"""
# 	if length == -1:
# 		length = len(data)
	#cov = (secondMoment/length-np.outer(mean, mean))
	cov = (secondMoment-np.outer(mean, mean))
	if besselsCorrection != 0:
		cov *= (length/(length-1.0))
	eg, ev = LA.eigh(cov)
	eg2 = []
	ev2 = []
#	print "+++sph eg+++"
#	print eg
#	print "----"
	for i in range(0, len(eg)):
		if (eg[i] >= threshold):
			eg2.append(1.0/np.sqrt(eg[i]))
			ev2.append(ev.T[i])
	return np.dot(np.transpose(ev2), np.diag(eg2))

def calcSpheringParametersAndDataRefImp(data, threshold = 0.00001, offset = 0, length = -1, besselsCorrection = 0):
	"""
	Returns sphered version of the given data and the sphering parameters as well.
	The data is expected in the format (time, dims). The sphering matrix may reduce
	the dimensionality of the data by deleting dimensions that have (near) zero
	variance. The returned sphering matrix has the shape (high dim, low dim).
	The sphering is done with respect to a covariance-matrix including Bessel's correction,
	if the besselsCorrection parameter is True (default is False).
	"""
	if besselsCorrection != 0:
		besselsCorrection = 1
	if length == -1:
		length = len(data)
	mean = data[offset:].mean(0)
	mean2 = data[offset:].sum(0)/len(data[offset:])
	data0 = data-np.outer(np.ones(len(data)), mean)
	#With Bessel's correction:
	#cov = np.multiply(np.dot(data0[offset:].T, data0[offset:]), 1.0/(len(data)-offset-besselsCorrection))
	cov = np.dot(data0[offset:].T, data0[offset:])
	eg, ev = LA.eigh(cov)
	eg2 = []
	ev2 = []
	for i in range(0, len(eg)):
		if (eg[i] >= threshold):
			#eg2.append(1.0/(np.sqrt(eg[i]*len(data))))
			eg2.append(1.0/(np.sqrt(eg[i]/(len(data)-offset-besselsCorrection))))
			ev2.append(ev.T[i])
	S = np.dot(np.transpose(ev2), np.diag(eg2))
#	for i in range(0, len(eg)):
#		eg[i] = eg[i]**(-0.5)
#	S = np.dot(ev.T, np.diag(eg))
	return [mean, S, data0.dot(S)]

#def calcAutoCorrelationList(data, p, besselsCorrection = 0):
#	er = []
#	data0 = data[p:].T
#	for i in range(1, p+1):
#		#er.append(np.multiply(np.dot(data0, data[p-i:len(data)-i]), 1.0/(len(data)-p)))
#		er.append(np.dot(data0, data[p-i:len(data)-i]))
#	return er
#
##	if besselsCorrection != 0:
##		besselsCorrection = 1
###	data0 = data[p:].T
##	for i in range(1, p+1):
###		er.append(np.dot(data0, data[p-i:len(data)-i]))
##		er.append(np.multiply(np.dot(data[i:].T, data[:-i]), (1.0*(len(data)-besselsCorrection))/(len(data)-i-besselsCorrection)))
##	return er
#
#def zZiFromAutoCorrelationsList(auto, i, p):
#	return np.hstack(auto[i:i+p])
#
#def ZZFromAutoCorrelationsList(auto, p, data, besselsCorrection = 0, dataSphered = True):
#	if besselsCorrection != 0:
#		besselsCorrection = 1
#	cov = np.multiply(np.identity(len(data[0])), len(data)-besselsCorrection)
#	if not dataSphered:
#		#cov = np.multiply(np.dot(data.T, data), 1.0/(len(data)-besselsCorrection))
#		cov = np.dot(data.T, data)
#	lines = []
#	#i iterates through lines, j through columns
#	for i in range(p):
#		line = []
#		for j in range(p):
#			if i == j:
#				line.append(cov)
#			elif i > j:
#				line.append(lines[j][i].T)
#			else:
#				line.append(auto[i-1])
#		lines.append(line)
#	for i in range(p):
#		lines[i] = np.hstack(lines[i])
#	return np.vstack(lines)

def invertByProjectionRefImp(M, evThreshold = 0.000001):
	eg, ev = LA.eigh(M)
	r = evThreshold**2
	for i in range(0, len(eg)):
		if (eg[i]**2 > r):
			eg[i] = 1.0/eg[i]
		else:
			eg[i] = 0.0
	return np.dot(ev, np.dot(np.diag(eg), ev.T))

def linearRegressionCoeff(srcData, destData):
	srcCov = np.dot(srcData.T, srcData)
	cor = np.dot(srcData.T, destData)
	srcCovI = invertByProjectionRefImp(srcCov)
	return np.dot(srcCovI, cor)

# def affineRegressionCoeff(srcData, destData):
# 	srcCov = np.dot(srcData.T, srcData)
# 	cor = np.dot(srcData.T, destData)
# 	srcCovI = invertByProjectionRefImp(srcCov)
# 	return np.dot(srcCovI, cor)

#def fullRegressionMatrixFromRegressionCoeff(W):
#	tmp = np.zeros([len(W[0])-len(W), len(W[0])])
#	for i in range(0, len(tmp)):
#		tmp[i][i] = 1.0
#	return np.vstack([W, tmp])

def calcShiftedDataListRefImp(data, p, p_offset = 1):
	dataZeta = []
	off = p_offset
	if off == 0:
		off += 1
		dataZeta.append(data[p:])
	for i in range(off, p+1):
		dataZeta.append(data[p-i: -i])
	return dataZeta

def calcZetaDataRefImp(data, p, delay = 0):
	return np.hstack(calcShiftedDataListRefImp(data, p+delay, 1+delay))
#	dataZeta = calcShiftedDataList(data, p-1)
#	return np.hstack(dataZeta)[0:len(dataZeta[0])-1-delay]

def calcZetacDataRefImp(data, p, delay = 0):
	return np.hstack([calcZetaDataRefImp(data, p, delay), np.ones([len(data)-p-delay, 1])])

def calcZeta0DataRefImp(data, p):
	return np.hstack(calcShiftedDataListRefImp(data[1:], p-1, 0))

def calcZeta0cDataRefImp(data, p):
	return np.hstack([calcZeta0DataRefImp(data, p), np.ones([len(data)-p, 1])])

def empiricalRawErrorRefImp(data, W, srcData = None):
	"""
	Measures how good following equation is fulfilled in average over time:  z = W zeta
	Evaluates data from p to len(data), retrieves p via len(W[0])/len(W)
	"""
	sDat = srcData
	if sDat is None:
		sDat = data
	p = len(W[0])/len(W)
	pre = np.dot(calcZetaDataRefImp(sDat, p), W.T)
	#real = data[0:len(data)-p]
	real = data[p:len(data)]
	#err = (pre[0]-real[0])**2
	#for i in range(1, len(real)):
	#	err += (pre[i]-real[i])**2
	#return np.multiply(err, 1.0/len(real))
	return (LA.norm(real-pre)**2)/len(real)

def empiricalRawErrorAffineRefImp(data, W_c, srcData = None):
	"""
	Measures how good following equation is fulfilled in average over time:  z = W zeta + c
	Evaluates data from p to len(data), retrieves p via len(W[0])/len(W)
	"""
	sDat = srcData
	if sDat is None:
		sDat = data
	p = (len(W_c[0])-1)/len(W_c)
	pre = np.dot(calcZetacDataRefImp(sDat, p), W_c.T)
	real = data[p:len(data)]
	return (LA.norm(real-pre)**2)/len(real)

def empiricalRawErrorComponentsRefImp(data, W, srcData = None):
	"""
	Measures how good following equation is fulfilled in average over time:  z = W zeta
	Evaluates data from p to len(data), retrieves p via len(W[0])/len(W)
	"""
	sDat = srcData
	if sDat is None:
		sDat = data
	p = len(W[0])/len(W)
	pre = np.dot(calcZetaDataRefImp(sDat, p), W.T)
	real = data[p:len(data)]
	res = (pre-real).T
	er = [] #np.zeros(len(res))
	for i in range(0, len(res)):
		er.append(np.inner(res[i], res[i]))
	return np.multiply(er, 1.0/len(real))

def empiricalRawErrorComponentsAffineRefImp(data, W_c, srcData = None):
	"""
	Measures how good following equation is fulfilled in average over time:  z = W zeta + c
	Evaluates data from p to len(data), retrieves p via len(W[0])/len(W)
	"""
	sDat = srcData
	if sDat is None:
		sDat = data
	p = (len(W_c[0])-1)/len(W_c)
	pre = np.dot(calcZetacDataRefImp(sDat, p), W_c.T)
	real = data[p:len(data)]
	res = (pre-real).T
	er = [] #np.zeros(len(res))
	for i in range(0, len(res)):
		er.append(np.inner(res[i], res[i]))
	return np.multiply(er, 1.0/len(real))

def predictNextFeatures(data, W):
	p = len(W[0])/len(W)
	return np.dot(np.hstack(data[-p:][::-1]), W.T)

def predictNextFeaturesAffine(data, W_c):
	p = (len(W_c[0])-1)/len(W_c)
	return np.dot(np.hstack(data[-p:][::-1]), W_c.T[:-1])+W_c.T[-1]

def calcRegressionCoeffRefImp(data, p, evThreshold = 0.0001):
	zeta = calcZetaDataRefImp(data, p)
	z_p = data[p:]
	zZT = np.dot(z_p.T, zeta)
	ZZT = np.dot(zeta.T, zeta)
	ZZTI = invertByProjectionRefImp(ZZT, evThreshold)
	return np.dot(zZT, ZZTI)

def calcRegressionCoeffAffineRefImp(data, p, evThreshold = 0.0001):
	zetac = calcZetacDataRefImp(data, p)
	z_p = data[p:]
	zZT = np.dot(z_p.T, zetac)
	ZZT = np.dot(zetac.T, zetac)
	ZZTI = invertByProjectionRefImp(ZZT, evThreshold)
	return np.dot(zZT, ZZTI)

def buildV_cRefImp(W_c):
	p = (len(W_c[0])-1)/len(W_c)
	id = np.identity((p-1)*len(W_c))
	zer = np.zeros([len(id), len(W_c)+1])
	bot = np.zeros([len(W_c[0])])
	bot[len(bot)-1] = 1.0
	return np.vstack([W_c, np.hstack([id, zer]), bot])

def buildVRefImp(W):
	p = (len(W[0]))/len(W)
	id = np.identity((p-1)*len(W))
	zer = np.zeros([len(id), len(W)])
	return np.vstack([W, np.hstack([id, zer])])

def calcExtractionForErrorCovRefImp(X, r):
	eg, ev = LA.eigh(X)
	sort = np.argsort(eg)
	ev2 = []
	for i in range(0, r):
		ev2.append(ev.T[sort[i]])
	return np.transpose(ev2)

def calcExtractionWithWeightsForErrorCovRefImp(X, r):
	eg, ev = LA.eigh(X)
	sort = np.argsort(eg)
	ev2 = []
	eg2 = []
	for i in range(0, r):
		ev2.append(ev.T[sort[i]])
		eg2.append(eg[sort[i]])
	return np.transpose(ev2), np.array(eg), np.array(eg2)

def calcErrorCoeffRefImp(data, W, k = 0):
	p = (len(W[0]))/len(W)
	V = buildVRefImp(W)
	X = np.zeros([len(W), len(W)])
	#z_pk = data[p+k:]
	WV = W
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V)
		zeta1_i = calcZetaDataRefImp(data, p, i)
		z_i_pre = np.dot(zeta1_i, WV.T)
		res_i = data[p+i:]-z_i_pre
		X += np.dot(res_i.T, res_i)
	return X

def calcErrorCoeffConstLenRefImp(data, W, k = 0):
	p = (len(W[0]))/len(W)
	V = buildVRefImp(W)
	X = np.zeros([len(W), len(W)])
	z_pk = data[p+k:]
	WV = W
	#print z_pk
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V)
		zeta1_i = calcZetaDataRefImp(data, p, i)[k-i:]
		#print zeta1_i
		z_i_pre = np.dot(zeta1_i, WV.T)
		res_i = z_pk-z_i_pre
		X += np.dot(res_i.T, res_i)
	return X

def calcErrorCoeffAffineRefImp(data, W_c, k = 0):
	p = (len(W_c[0])-1)/len(W_c)
	V_c = buildV_cRefImp(W_c)
	X = np.zeros([len(W_c), len(W_c)])
	#z_pk = data[p+k:]
	WV_c = W_c
	for i in range(0, k+1):
		if i > 0:
			WV_c = np.dot(WV_c, V_c)
		zeta1_i_c = calcZetacDataRefImp(data, p, i)
		z_i_pre = np.dot(zeta1_i_c, WV_c.T)
		res_i = data[p+i:]-z_i_pre
		X += np.dot(res_i.T, res_i)
#		if i == 1:
#			#print np.dot(zeta1_i_c.T, zeta1_i_c)#/(len(zeta1_i_c))
#			print np.dot(data[p+i:].T, zeta1_i_c)
#			print zeta1_i_c.mean(0)*(len(zeta1_i_c))
#			print data[p-1:-1].mean(0)*(len(data)-p)
	return X

def calcErrorCoeffConstLenAffineRefImp(data, W_c, k = 0):
	p = (len(W_c[0])-1)/len(W_c)
	V_c = buildV_cRefImp(W_c)
	X = np.zeros([len(W_c), len(W_c)])
	z_pk = data[p+k:]
	WV_c = W_c
	#print z_pk
	for i in range(0, k+1):
		if i > 0:
			WV_c = np.dot(WV_c, V_c)
		zeta1_i_c = calcZetacDataRefImp(data, p, i)[k-i:]
		#print zeta1_i
		z_i_pre = np.dot(zeta1_i_c, WV_c.T)
		res_i = z_pk-z_i_pre
		X += np.dot(res_i.T, res_i)
#		if i == 1:
#			print np.dot(zeta1_i_c.T, zeta1_i_c)#/(len(zeta1_i_c))
			#print np.dot(data[p+i:].T, zeta1_i_c)
#			print zeta1_i_c.mean(0)*(len(zeta1_i_c))
#			print data[p-1:-1].mean(0)*(len(data)-p)
	return X

def calcErrorCoeff(data, W, k = 0):
	p = len(W[0])/len(W)
	X = np.zeros([len(W), len(W)])
	corList = [np.dot(data[p:].T, data[p:])]

	for i in range(1, p+1):
		corList.append(np.dot(data[p:].T, data[p-i:-i]))

	zetaList = []
	lastLine = []
	zetaBlockList = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]+np.outer(data[p-1], data[p-1-j])-np.outer(data[-1], data[-1-j]))
				else:
					zetaLine.append(lastLine[j-1]+np.outer(data[p-1-i], data[p-1-j])-np.outer(data[-1-i], data[-1-j]))

		zetaList.append(zetaLine)
		lastLine = zetaLine

	cov_p = np.dot(data[p:].T, data[p:])
	WV = W
	V = buildVRefImp(W)
	for i in range(k+1):
		if i > 0:
			WV = np.dot(WV, V)
		if i > 0:
			for j in range(i+1, len(corList)):
				corList[j] -= np.outer(data[p+i-1], data[p-j+i-1])
			corList.append(np.dot(data[p+i:].T, data[:-p-i]))
		zZi = np.hstack(corList[1+i:])
		if i == 0:
			for l in range(p):
				zetaBlockList.append(np.hstack(zetaList[l]))
		else:
			for l in range(p):
				for j in range(p):
					if j < l:
						zetaList[l][j] = zetaList[j][l].T
					else:
						zetaList[l][j] -= np.outer(data[-(i+1)-l], data[-(i+1)-j])
			for l in range(p):
				zetaBlockList[l] = np.hstack(zetaList[l])
		ZZi = np.vstack(zetaBlockList)
		K = np.dot(zZi, WV.T)
		X += cov_p -K-K.T + np.dot(np.dot(WV, ZZi), WV.T)
		if i < k:
			cov_p -= np.outer(data[p+i], data[p+i])
	return X

def calcErrorCoeffConstLen(data, W, k = 0):
	p = len(W[0])/len(W)
	z_pk = data[p+k:]
	corList = [np.dot(z_pk.T, z_pk)]
	for i in range(1, k+p+1):
		corList.append(np.dot(z_pk.T, data[p+k-i:-i]))
	return calcErrorCoeffConstLenFromAutoCorrelations(W, corList, data[:p+k], data[-p-k:], k)

def calcErrorCoeffConstLenFromAutoCorrelations(W, corList, startData, endData, k = 0):
	p = len(corList)-1-k
	zetaList = []
	lastLine = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]-np.outer(endData[-1], endData[-1-j])+np.outer(startData[p+k-1], startData[p+k-1-j]))
				else:
					zetaLine.append(lastLine[j-1]-np.outer(endData[-1-i], endData[-1-j])+np.outer(startData[p+k-1-i], startData[p+k-1-j]))
		zetaList.append(zetaLine)
		lastLine = zetaLine
	for i in range(p):
		zetaList[i] = np.hstack(zetaList[i])
	ZZi = np.vstack(zetaList)
	return calcErrorCoeffConstLenFromCorrelations(W, np.hstack(corList[1:p+1]), ZZi, lastLine, corList, startData, endData, k)
	
def calcErrorCoeffConstLenFromCorrelations(W, zZ, ZZ, lastLine, corList, startData, endData, k = 0):
	p = len(corList)-1-k
	WV = W
	V = buildVRefImp(W)
	Y = corList[0]*(k+1)
	pnl = len(W)*(p-1)
	zZi = zZ
	ZZi = ZZ
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V)
			zZi = np.hstack(corList[1+i:p+i+1])
		K = np.dot(zZi, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi), WV.T)
		if i < k:
			for j in range(p):
				lastLine[j] += -np.outer(endData[-p-i-1], endData[-j-2-i])+np.outer(startData[k-1-i], startData[p-i+k-j-2])
			line = np.hstack(lastLine)
			ZZi = np.vstack([np.hstack([ ZZi[len(W):].T[len(W):], line.T[:pnl]]), line])
	return Y

#def calcErrorCoeffConstLenFromCorrelations2(W, zZ, ZZ, lastLine, corList, S, start_cor, end_cor, k = 0):
#def calcErrorCoeffConstLenFromCorrelations2(W, zZ, ZZ, lastLine, corList, startData, endData, S, start_cor, end_cor, k = 0):
def calcErrorCoeffConstLenFromCorrelations2(W, zZ, ZZ, lastLine, corList, S, start_cor, end_cor, k = 0):
	p = len(corList)-1-k
	WV = W
	V = buildVRefImp(W)
	Y = corList[0]*(k+1)
	pnl = len(W)*(p-1)
	zZi = zZ
	ZZi = ZZ
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V)
			zZi = np.hstack(corList[1+i:p+i+1])
		K = np.dot(zZi, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi), WV.T)
		if i < k:
			for j in range(p):
				#lastLine[j] += -np.outer(endData[-p-i-1], endData[-j-2-i])+np.outer(startData[-p-1-i], startData[-i-j-2])
				lastLine[j] += np.dot(S.T, np.dot(-end_cor[-p-i-1][-j-2-i]+start_cor[-p-1-i][-i-j-2], S))
			line = np.hstack(lastLine)
			ZZi = np.vstack([np.hstack([ ZZi[len(W):].T[len(W):], line.T[:pnl]]), line])
	return Y

def calcErrorCoeffConstLenRoll(data, W, k = 0):
	p = len(W[0])/len(W)
	z_pk = data[p+k:]
	WV = W
	V = buildVRefImp(W)
	Y = np.dot(z_pk.T, z_pk)*(k+1)
	corList = [np.dot(z_pk.T, z_pk)]
	for i in range(1, k+p+1):
		corList.append(np.dot(z_pk.T, data[p+k-i:-i]))

	zetaList = []
	lastLine = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]-np.outer(data[-1], data[-1-j])+np.outer(data[p+k-1], data[p+k-1-j]))
				else:
					zetaLine.append(lastLine[j-1]-np.outer(data[-1-i], data[-1-j])+np.outer(data[p+k-1-i], data[p+k-1-j]))
		zetaList.append(zetaLine)
		lastLine = zetaLine
	for i in range(p):
		zetaList[i] = np.hstack(zetaList[i])
	ZZi = np.vstack(zetaList)

	pnl = len(W)*(p-1)
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V)
		zZi = np.hstack(corList[1+i:p+i+1])
		K = np.dot(zZi, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi), WV.T)
		if i < k:
			ZZi = np.roll(np.roll(ZZi, len(W), 0), len(W), 1)
			for j in range(p):
				lastLine[j] += -np.outer(data[-p-i-1], data[-j-2-i])+np.outer(data[k-1-i], data[p-i+k-j-2])
				ZZi[pnl:, len(W)*j:len(W)*(j+1)] = lastLine[j]
				if j != p:
					ZZi[len(W)*j:len(W)*(j+1), pnl:] = lastLine[j].T
#			line = np.hstack(lastLine)
#			for j in range(p):
#				lastLine[j] += -np.outer(data[-p-i-1], data[-j-2-i])+np.outer(data[k-1-i], data[p-i+k-j-2])
#			line = np.hstack(lastLine)
#			ZZi = np.vstack([np.hstack([ ZZi[pnl:].T[pnl:], line.T[:pnl]]), line])
#			print "ZZi"
#			print ZZi
#			print ZZi2

	return Y

def calcErrorCoeffAffine(data, W_c, k = 0):
	p = (len(W_c[0])-1)/len(W_c)
	X = np.zeros([len(W_c), len(W_c)])
	corList = [np.dot(data[p:].T, data[p:])]

	m = data[p:].mean(0)*(len(data)-p)
	meanList = [m]

	for i in range(1, p+1):
		corList.append(np.dot(data[p:].T, data[p-i:-i]))
		meanList.append(meanList[i-1] - data[-i]+data[p-i])
		#meanList[i] -= data[i-1]
	#print np.hstack(meanList[1:])

	zetaList = []
	lastLine = []
	zetaBlockList = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]+np.outer(data[p-1], data[p-1-j])-np.outer(data[-1], data[-1-j]))
				else:
					zetaLine.append(lastLine[j-1]+np.outer(data[p-1-i], data[p-1-j])-np.outer(data[-1-i], data[-1-j]))

		zetaList.append(zetaLine)
		lastLine = zetaLine

	cov_p = np.dot(data[p:].T, data[p:])
	WV = W_c
	V_c = buildV_cRefImp(W_c)
	for i in range(k+1):
		if i > 0:
			WV = np.dot(WV, V_c)
		if i > 0:
			meanList[0] -= data[p+i-1]
			for j in range(i+1, len(corList)):
				corList[j] -= np.outer(data[p+i-1], data[p-j+i-1])
				meanList[j] -= data[p-j+i-1]
			corList.append(np.dot(data[p+i:].T, data[:-p-i]))
			#corList.append(np.dot(data[p:].T, data[p-i:-i]))
			#meanList.append(meanList[-1] - data[-p-i])
			meanList.append(data[:-p-i].mean(0)*len(data[:-p-i])) #todo: calc this from meanList[-1]

		zZi_c = np.vstack([np.hstack(corList[1+i:]).T, meanList[0]]).T
		if i == 0:
			for l in range(p):
				zetaBlockList.append(np.hstack(zetaList[l]))
		else:
			for l in range(p):
				for j in range(p):
					if j < l:
						zetaList[l][j] = zetaList[j][l].T
					else:
						zetaList[l][j] -= np.outer(data[-(i+1)-l], data[-(i+1)-j])
			for l in range(p):
				zetaBlockList[l] = np.hstack(zetaList[l])
		ZZi = np.vstack(zetaBlockList)
		ml = np.hstack(meanList[1+i:])
		ZZi_c = np.vstack([ZZi, ml]).T
		ml = np.hstack([ml, [len(data)-p-i]])
		ZZi_c = np.vstack([ZZi_c, ml])
#		if i == 1:
#			print zZi_c
		K = np.dot(zZi_c, WV.T)
		X += cov_p -K-K.T + np.dot(np.dot(WV, ZZi_c), WV.T)
		if i < k:
			cov_p -= np.outer(data[p+i], data[p+i])
	return X

def calcErrorCoeffConstLenAffine(data, W_c, k = 0):
	p = (len(W_c[0])-1)/len(W_c)
	z_pk = data[p+k:]
	corList = [np.dot(z_pk.T, z_pk)]
	meanList = [z_pk.mean(0)*len(z_pk)]
	for i in range(1, k+p+1):
		corList.append(np.dot(z_pk.T, data[p+k-i:-i]))
		meanList.append(meanList[-1]-data[-i]+data[p+k-i])
	return calcErrorCoeffConstLenAffineFromAutoCorrelations(W_c, corList, meanList, len(data), data[:p+k], data[-p-k:], k)
	
def calcErrorCoeffConstLenAffineFromAutoCorrelations(W_c, corList, meanList, dataLen, startData, endData, k = 0):
	zetaList = []
	lastLine = []
	p = len(corList)-1-k
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]-np.outer(endData[-1], endData[-1-j])+np.outer(startData[p+k-1], startData[p+k-1-j]))
				else:
					zetaLine.append(lastLine[j-1]-np.outer(endData[-1-i], endData[-1-j])+np.outer(startData[p+k-1-i], startData[p+k-1-j]))
		zetaList.append(zetaLine)
		lastLine = zetaLine
	for i in range(p):
		zetaList[i] = np.hstack(zetaList[i])
	ml = np.hstack(meanList[1:p+1])
	print ml
	ZZi_c = np.vstack([np.vstack(zetaList), ml])
	ml1 = np.hstack([ml, [dataLen-p-k]])
	ZZi_c = np.vstack([ZZi_c.T, ml1]).T
	return calcErrorCoeffConstLenAffineFromCorrelations(W_c, np.vstack([np.hstack(corList[1:p+1]).T, meanList[0]]).T, ZZi_c, lastLine, corList, meanList, dataLen, startData, endData, k)

def calcErrorCoeffConstLenAffineFromCorrelations(W_c, zZ_c, ZZ_c, lastLine, corList, meanList, dataLen, startData, endData, k = 0):
	#print ZZi_c
	p = len(corList)-1-k
	WV = W_c
	V_c = buildV_cRefImp(W_c)
	#Y = np.dot(z_pk.T, z_pk)*(k+1)
	Y = corList[0]*(k+1)
	pnl = len(W_c)*(p-1)
	ZZi_c = ZZ_c
	zZi_c = zZ_c
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V_c)
			zZi_c = np.vstack([np.hstack(corList[1+i:p+i+1]).T, meanList[0]]).T
#		if i == 1:
#			print ZZi_c
		K = np.dot(zZi_c, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi_c), WV.T)
		if i < k:
			for j in range(p):
				lastLine[j] += -np.outer(endData[-p-i-1], endData[-j-2-i])+np.outer(startData[k-1-i], startData[p-i+k-j-2])
			line = np.hstack(lastLine)
			ZZi_c = np.vstack([np.hstack([ ZZi_c[len(W_c):-1].T[len(W_c):-1], line.T[:pnl]]), line])
			ml = np.hstack(meanList[i+2:p+i+2])
			ZZi_c = np.vstack([ZZi_c, ml])
			ml1 = np.hstack([ml, [dataLen-p-k]])
			ZZi_c = np.vstack([ZZi_c.T, ml1]).T
	return Y

#def calcErrorCoeffConstLenAffineFromCorrelations2(W_c, zZ_c, ZZ_c, lastLine, corList, meanList, dataLen, startData, endData, k = 0):
def calcErrorCoeffConstLenAffineFromCorrelations2(W_c, zZ_c, ZZ_c, lastLine, corList, meanList, dataLen, S, start_cor, end_cor, k = 0):
	#print ZZi_c
	p = len(corList)-1-k
	WV = W_c
	V_c = buildV_cRefImp(W_c)
	#Y = np.dot(z_pk.T, z_pk)*(k+1)
	Y = corList[0]*(k+1)
	pnl = len(W_c)*(p-1)
	ZZi_c = ZZ_c
	zZi_c = zZ_c
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V_c)
			zZi_c = np.vstack([np.hstack(corList[1+i:p+i+1]).T, meanList[0]]).T
#		if i == 1:
#			print ZZi_c
		K = np.dot(zZi_c, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi_c), WV.T)
		if i < k:
			for j in range(p):
				#lastLine[j] += -np.outer(endData[-p-i-1], endData[-j-2-i])+np.outer(startData[-p-1-i], startData[-i-j-2])
				lastLine[j] += np.dot(S.T, np.dot(-end_cor[-p-i-1][-j-2-i]+start_cor[-p-1-i][-i-j-2], S))
			line = np.hstack(lastLine)
			#print pnl
			#print len(W_c)
			#print ZZi_c[len(W_c):-1].T[len(W_c):-1].shape
			#print line.T[:pnl].shape
			a = np.hstack([ ZZi_c[len(W_c):-1].T[len(W_c):-1], line.T[:pnl]])
			ZZi_c = np.vstack([a, line])
			ml = np.hstack(meanList[i+2:p+i+2])
			ZZi_c = np.vstack([ZZi_c, ml])
			ml1 = np.hstack([ml, [dataLen-p-k]])
			ZZi_c = np.vstack([ZZi_c.T, ml1]).T
	return Y

def calcErrorCoeffConstLenAffineRoll(data, W_c, k = 0):
	p = (len(W_c[0])-1)/len(W_c)
	z_pk = data[p+k:]
	WV = W_c
	V_c = buildV_cRefImp(W_c)
	Y = np.dot(z_pk.T, z_pk)*(k+1)
	corList = [np.dot(z_pk.T, z_pk)]
	meanList = [z_pk.mean(0)*len(z_pk)]
	for i in range(1, k+p+1):
		corList.append(np.dot(z_pk.T, data[p+k-i:-i]))
		meanList.append(meanList[-1]-data[-i]+data[p+k-i])

	zetaList = []
	lastLine = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]-np.outer(data[-1], data[-1-j])+np.outer(data[p+k-1], data[p+k-1-j]))
				else:
					zetaLine.append(lastLine[j-1]-np.outer(data[-1-i], data[-1-j])+np.outer(data[p+k-1-i], data[p+k-1-j]))
		zetaList.append(zetaLine)
		lastLine = zetaLine
	for i in range(p):
		zetaList[i] = np.hstack(zetaList[i])
	ml = np.hstack(meanList[1:p+1])
#	print ml
	ZZi_c = np.vstack([np.vstack(zetaList), ml])
	ml1 = np.hstack([ml, [len(z_pk)]])
	ZZi_c = np.vstack([ZZi_c.T, ml1]).T

	#print ZZi_c

	pnl = len(W_c)*(p-1)
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, V_c)
		zZi_c = np.vstack([np.hstack(corList[1+i:p+i+1]).T, meanList[0]]).T
#		if i == 1:
#			print ZZi_c
		K = np.dot(zZi_c, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi_c), WV.T)
		if i < k:
			ZZi_c = np.roll(np.roll(ZZi_c[:-1, :-1], len(W_c), 0), len(W_c), 1)
			for j in range(p):
				lastLine[j] += -np.outer(data[-p-i-1], data[-j-2-i])+np.outer(data[k-1-i], data[p-i+k-j-2])
				ZZi_c[pnl:, len(W_c)*j:len(W_c)*(j+1)] = lastLine[j]
				if j != p:
					ZZi_c[len(W_c)*j:len(W_c)*(j+1), pnl:] = lastLine[j].T
#				ZZi_c2[-1:, len(W_c)*j:len(W_c)*(j+1)] = meanList[i+2+j]

#			for j in range(p):
#				lastLine[j] += -np.outer(data[-p-i-1], data[-j-2-i])+np.outer(data[k-1-i], data[p-i+k-j-2])
#			line = np.hstack(lastLine)
#			ZZi_c = np.vstack([np.hstack([ ZZi_c[pnl:-1].T[pnl:-1], line.T[:pnl]]), line])
			ml = np.hstack(meanList[i+2:p+i+2])
			ZZi_c = np.vstack([ZZi_c, ml])
			ml1 = np.hstack([ml, [len(z_pk)]])
			ZZi_c = np.vstack([ZZi_c.T, ml1]).T

#			print "ZZi_c"
#			print ZZi_c
#			print ZZi_c2
	return Y

def createTestData():
	#data = np.outer(range(0, 9), [1, 1.1, 1.11])
	data = np.outer(range(0, 12), [1, 2.0])#, 3.0])
	#place some ouliers:
	#data[3][1] = 1.0
	#data[5][2] = 1.0
	data[5][1] = 1.0
	data[4][1] = -1.0
	data[7][0] = -2.0
	data[7][0] = -3.0
	data[10][0] = -4.5
	data[11][1] = -6.0
	return data

def PFAReferenceImplementationTest():
	data = createTestData()
	p = 2
	print data
	print "------------------Sphering------------------"
	#Sphering:
	mean, S, z = calcSpheringParametersAndDataRefImp(data)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
	#Test Sphering:
	print z
	print np.multiply(np.dot(z.T, z), 1.0/len(data))

	print "------------------Fitting------------------"
	#Fitting:
	zeta = calcZetaDataRefImp(z, p)
	z_p = z[p:]
	W = calcRegressionCoeffRefImp(z, p)
	#Test fitting:
	#print W
	pre = np.dot(zeta, W.T) #np.dot(W, zeta.T).T
	res = z_p-pre
	print np.trace(np.dot(res.T, res))/len(res)
	print empiricalRawErrorRefImp(z, W)
	errComp = empiricalRawErrorComponentsRefImp(z, W)
	print errComp
	print np.sum(errComp)

	print "------------------PCA on Error Covariance------------------"
	r = 1
	#PCA on error covariance:
	X = np.dot(res.T, res)
	print X
	Ar = calcExtractionForErrorCovRefImp(X, r)
	m = np.dot(z, Ar)
	Wm = calcRegressionCoeffRefImp(m, p)
	print m
	errComp_m = empiricalRawErrorComponentsRefImp(m, Wm)
	print errComp_m
	print np.sum(errComp_m)
	print np.dot(m.T, m)/len(m)

	print "--------------------------------------------------"
	print "------------------Affine variant------------------"
	print "--------------------------------------------------"
	print "------------------Sphering------------------"
	print "like before"

	print "------------------Fitting------------------"
	#Fitting:
	zeta_c = calcZetacDataRefImp(z, p)
#	print zeta_c

#	z_p = z[p:]
	W_c = calcRegressionCoeffAffineRefImp(z, p)
	#Test fitting:
	#print W
	pre_c = np.dot(zeta_c, W_c.T) #np.dot(W, zeta.T).T
	res_c = z_p-pre_c
	print np.trace(np.dot(res_c.T, res_c))/len(res_c)
	print empiricalRawErrorAffineRefImp(z, W_c)
	errComp_c = empiricalRawErrorComponentsAffineRefImp(z, W_c)
	print errComp_c
	print np.sum(errComp_c)
	print W_c

	print "------------------Fitting Sinus------------------"
	sig_x = lambda t: [np.sin(t)]
	ts = range(0, 50)
	sin_x = np.array([sig_x(t) for t in ts])
	#print sin_x
	sin_mean, sin_S, sin_z = calcSpheringParametersAndDataRefImp(sin_x)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
	#Test Sphering:
	#print sin_z
	print "sin_z cov:"
	print np.multiply(np.dot(sin_z.T, sin_z), 1.0/len(sin_z))

	sin_p = 2
	sin_zeta_c = calcZetacDataRefImp(sin_z, sin_p)
#	print zeta_c

	sin_z_p = sin_z[sin_p:]
	sin_W_c = calcRegressionCoeffAffineRefImp(sin_z, sin_p)
	#Test fitting:
	#print W
	sin_pre_c = np.dot(sin_zeta_c, sin_W_c.T) #np.dot(W, zeta.T).T
	sin_res_c = sin_z_p-sin_pre_c
	print "sin_z, sin_W_c error:"
	print np.trace(np.dot(sin_res_c.T, sin_res_c))/len(sin_res_c)
	print empiricalRawErrorAffineRefImp(sin_z, sin_W_c)
	sin_errComp_c = empiricalRawErrorComponentsAffineRefImp(sin_z, sin_W_c)
	print sin_errComp_c
	print np.sum(sin_errComp_c)
	print "fitted sin W_c:"
	print sin_W_c
	print "analytic sin W_c:"
	cs = np.cos(1.0)
	sin_W_c_2 = np.array([[2*cs, -1.0, (-2.0+2.0*cs)*sin_S[0][0]*sin_mean[0]]])
	print empiricalRawErrorAffineRefImp(sin_z, sin_W_c_2)
	print sin_W_c_2

	print "------------------PCA on Error Covariance------------------"
	r_c = 1
	#PCA on error covariance:
	X_c = np.dot(res_c.T, res_c)
	print X_c
	Ar_c = calcExtractionForErrorCovRefImp(X_c, r_c)
	m_c = np.dot(z, Ar_c)
	Wm_c = calcRegressionCoeffAffineRefImp(m_c, p)
	print m_c
	print empiricalRawErrorAffineRefImp(m_c, Wm_c)
	print np.dot(m_c.T, m_c)/len(m_c)

	print "------------------PCA on Error Covariance - Noisy sine------------------"
	sig_x2 = lambda t: [np.sin(t), np.random.rand(1)[0]]
	#ts = range(0, 250)
	nsin_x = np.array([sig_x2(t) for t in ts])
	nsin_mean, nsin_S, nsin_z = calcSpheringParametersAndDataRefImp(nsin_x)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
	#Test Sphering:
	#print sin_z
	print "nsin_z cov:"
	print np.multiply(np.dot(nsin_z.T, nsin_z), 1.0/len(nsin_z))

	nsin_p = 2
	nsin_zeta_c = calcZetacDataRefImp(nsin_z, nsin_p)
##	print zeta_c

	nsin_z_p = nsin_z[nsin_p:]
	nsin_W_c = calcRegressionCoeffAffineRefImp(nsin_z, nsin_p)
	#Test fitting:
	#print W
	nsin_pre_c = np.dot(nsin_zeta_c, nsin_W_c.T) #np.dot(W, zeta.T).T
	nsin_res_c = nsin_z_p-nsin_pre_c
	print "nsin_z, nsin_W_c error:"
	print np.trace(np.dot(nsin_res_c.T, nsin_res_c))/len(nsin_res_c)
	print empiricalRawErrorAffineRefImp(nsin_z, nsin_W_c)
	nsin_errComp_c = empiricalRawErrorComponentsAffineRefImp(nsin_z, nsin_W_c)
	print nsin_errComp_c
	print np.sum(nsin_errComp_c)
#	print "fitted sin W_c:"
#	print nsin_W_c
#	print "analytic sin W_c:"
#	cs = np.cos(1.0)
#	nsin_W_c_2 = np.array([[2*cs, -1.0, (-2.0+2.0*cs)*nsin_S[0][0]*nsin_mean[0]]])
#	print empiricalRawErrorAffine(nsin_z, nsin_W_c_2)
#	print nsin_W_c_2

	#print "------------------PCA on Error Covariance------------------"
	nsin_r_c = 1
	#PCA on error covariance:
	nsin_X_c = np.dot(nsin_res_c.T, nsin_res_c)
	print nsin_X_c
	nsin_Ar_c = calcExtractionForErrorCovRefImp(nsin_X_c, nsin_r_c)
	nsin_m_c = np.dot(nsin_z, nsin_Ar_c)
	nsin_Wm_c = calcRegressionCoeffAffineRefImp(nsin_m_c, nsin_p)
	#print nsin_m_c
	print empiricalRawErrorAffineRefImp(nsin_m_c, nsin_Wm_c)
	print np.dot(nsin_m_c.T, nsin_m_c)/len(nsin_m_c)
	print "fitted nsin Wm_c:"
	print nsin_Wm_c
	print "analytic nsin Wm_c:"
	SI = LA.inv(nsin_S.T)
	factor = LA.norm(np.dot(np.array([1.0, 0]), SI))
	Ar2 = np.multiply(np.dot(np.array([1.0, 0]), SI), 1.0/factor)
	nsin_Wm_c_2 = np.array([[2*cs, -1.0, (-2.0+2.0*cs)*(1.0/factor)*nsin_mean[0]]])
	print empiricalRawErrorAffineRefImp(nsin_m_c, nsin_Wm_c_2)
	print nsin_Wm_c_2

	print ""
	print " _____________________________________________"
	print "/                                             \\"
	print "|---------------iteration stuff---------------|"
	print "\\_____________________________________________/"

	print ""
	#print zeta
	zeta0 = calcZeta0DataRefImp(z, p)
	#print zeta0
	ZZ0 = np.dot(zeta0.T, zeta)
	zZ = np.dot(z_p.T, zeta)
	ZZ = np.dot(zeta.T, zeta)
	ZZI = invertByProjectionRefImp(ZZ)
	W0 = np.dot(zZ, ZZI)
	V = np.dot(ZZ0, ZZI)
	print W0
	print W
	print ""
	print V
	print buildVRefImp(W)

	print ""
	print np.dot(ZZ, ZZI)

	print "calc iterated error coeff:"
	X_k = calcErrorCoeffRefImp(z, W, 1)
	Y_k = calcErrorCoeffConstLenRefImp(z, W, 1)
	print X
	print X_k
	print Y_k

	print "-------------affine version---------------"
	print ""
	#print zeta
	zeta0_c = calcZeta0cDataRefImp(z, p)
#	print zeta0_c
#	print zeta_c
	ZZ0_c = np.dot(zeta0_c.T, zeta_c)
	zZ_c = np.dot(z_p.T, zeta_c)
	ZZ_c = np.dot(zeta_c.T, zeta_c)
	ZZI_c = invertByProjectionRefImp(ZZ_c)
	W0_c = np.dot(zZ_c, ZZI_c)
	#The following idea of V_c is cumbersome, because it
	#puts error on the constant-one component of zeta_c,
	#if this reduces the overall error.
	#But the constant component may not be compromized in any case, so
	#this way to compute V_c is misleading:
#	V_c_cumb = np.dot(ZZ0_c, ZZI_c)
#	print V_c_cumb

	print W0_c
	print W_c
	print ""
	#Better way:
	V_c = buildV_cRefImp(W_c)
	print V_c
	print "calc iterated error coeff:"
	X_k_c = calcErrorCoeffAffineRefImp(z, W_c, 1)
	Y_k_c = calcErrorCoeffConstLenAffineRefImp(z, W_c, 1)
	print X_k
	print X_k_c
	print Y_k
	print Y_k_c

#todo: build unit-test from this:
def compareSphering():
	data = createTestData()
	p = 2
	print data
	print "------------------Sphering------------------"
	#Sphering:
	meanRef, SRef, zRef = calcSpheringParametersAndDataRefImp(data, besselsCorrection = 1)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
	#Test Sphering:
	#print z
	#print np.multiply(np.dot(z.T, z), 1.0/len(data))
	print SRef
	print meanRef

	#print data
	mean = data.mean(0)
	secondMoment = np.dot(data.T, data)
	print (secondMoment/len(data)-np.outer(mean, mean))
	S = calcSpheringMatrixFromMeanAndSecondMoment(mean, secondMoment, len(data), threshold = 0.0000001, besselsCorrection = 1)
	print S
	data0 = data-np.outer(np.ones(len(data)), mean)
	z = np.dot(data0, S)
	print zRef
	print z

def compareFitting():
	data = createTestData()
	p = 2
	print data
	print "------------------Sphering------------------"
	#Sphering:
	meanRef, SRef, zRef = calcSpheringParametersAndDataRefImp(data, besselsCorrection = 0)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
	print "------------------Fitting------------------"
	#Fitting:
	zeta = calcZetaDataRefImp(zRef, p)
	z_pRef = zRef[p:]
	WRef = calcRegressionCoeffRefImp(zRef, p)
	#print WRef
	ZZRef = np.dot(zeta.T, zeta)
	print ZZRef
	Z11Ref = np.dot(zRef[p-1:-1].T, zRef[p-1:-1])
	Z12Ref = np.dot(zRef[p-1:-1].T, zRef[p-2:-2])
	Z22Ref = np.dot(zRef[p-2:-2].T, zRef[p-2:-2])
	print Z11Ref
	#print Z12Ref
	#print Z22Ref
	print ""
	mean = data.mean(0)
	secondMoment = np.dot(data.T, data)
	S = calcSpheringMatrixFromMeanAndSecondMoment(mean, secondMoment, len(data), threshold = 0.0000001, besselsCorrection = 0)
	#x1 = data[p-1:-1]
	#x2 = data[p-2:-2]
	#X11 = np.dot(x1.T, x1)
	X00 = secondMoment-np.dot(data[0:p].T, data[0:p])
	X11 = X00-np.outer(data[-1], data[-1])+np.outer(data[p-1], data[p-1])
	#X12 = np.dot(x1.T, x2)
	X12 = np.dot(data[p-1:-1].T, data[p-2:-2])
	#X22 = np.dot(x2.T, x2)
	X22 = X11-np.outer(data[-2], data[-2])+np.outer(data[p-2], data[p-2])

	M = np.outer(mean, mean)*(len(data)-p)
	x0 = (mean*len(data)-data[0:p].mean(0)*p)
	xm1 = (x0 - data[-1]+data[p-1])
	xm2 = (xm1 - data[-2]+data[p-2])
	#xm1 = data[p-1:-1].mean(0)*(len(data)-p)
	#xm2 = data[p-2:-2].mean(0)*(len(data)-p)

	#z1_ = x1 - np.outer(np.ones(len(x1)), meanRef)
	#print np.dot(z1_, SRef)
	print Z11Ref
#	print np.dot(np.dot(x1 - np.outer(np.ones(len(x1)), meanRef), SRef).T, np.dot(x1 - np.outer(np.ones(len(x1)), meanRef), SRef))
#	print np.dot(np.dot(SRef.T, (x1 - np.outer(np.ones(len(x1)), meanRef)).T), np.dot(x1 - np.outer(np.ones(len(x1)), meanRef), SRef))

	#K = np.dot((x1 - np.outer(np.ones(len(x1)), meanRef)).T, x1 - np.outer(np.ones(len(x1)), meanRef))
	K11 = X11 +( - np.outer(xm1, mean) - np.outer(mean, xm1) + M)
	print np.dot(S.T, np.dot(K11, S))
	print ""
	print Z12Ref
	K12 = X12 +( - np.outer(xm1, mean) - np.outer(mean, xm2) + M)
	print np.dot(S.T, np.dot(K12, S))
	print ""
	print Z22Ref
	K22 = X22 +( - np.outer(xm2, mean) - np.outer(mean, xm2) + M)
	print np.dot(S.T, np.dot(K22, S))
	print ""

#	print ZZRef
#	print ZZRef[2:4].T[2:4]
	
#	print np.dot(x1.T, np.outer(np.ones(len(x1)), meanRef))
#	print np.outer(x1.mean(0)*len(x1), meanRef)

#	print np.dot(np.outer(np.ones(len(x1)), meanRef).T, x1)
#	print np.outer(meanRef, x1.mean(0)*len(x1))

#	print np.dot(np.outer(np.ones(len(x1)), meanRef).T, np.outer(np.ones(len(x1)), meanRef))
#	print np.outer(meanRef, meanRef)*len(x1)

#	x2 = data[p-2:-2]
#	X11 = np.dot(x1.T, x1)
#	X12 = np.dot(x1.T, x2)
#	X22 = np.dot(x2.T, x2)
#	xm1 = x1.mean(0)
#	xm2 = x2.mean(0)
#	K11 = np.outer(meanRef, meanRef) + (X11-np.outer(xm1, meanRef)-np.outer(meanRef, xm1))/(len(data)-p*1.0)
#	Z11 = np.dot(SRef.T, np.dot(K11, SRef))
#	print Z11
	

	#Test fitting:
	#print W
#	pre = np.dot(zeta, W.T) #np.dot(W, zeta.T).T
#	res = z_p-pre
#	print np.trace(np.dot(res.T, res))/len(res)
#	print empiricalRawErrorRefImp(z, W)
#	errComp = empiricalRawErrorComponentsRefImp(z, W)
#	print errComp
#	print np.sum(errComp)

def compareIteration():
	data = createTestData()
	p = 2
	print data
	print "------------------Sphering------------------"
	#Sphering:
	meanRef, SRef, zRef = calcSpheringParametersAndDataRefImp(data)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)

	print "------------------Fitting------------------"
	#Fitting:
	z_pRef = zRef[p:]
	zetaRef = calcZetaDataRefImp(zRef, p)
	WRef = calcRegressionCoeffRefImp(zRef, p)

	print "------------------PCA on Error Covariance------------------"
	r = 1
	#PCA on error covariance:
#	X = np.dot(res.T, res)
#	print X
#	Ar = calcExtractionForErrorCovRefImp(X, r)
#	m = np.dot(z, Ar)
#	Wm = calcRegressionCoeffRefImp(m, p)
#	print m
#	errComp_m = empiricalRawErrorComponentsRefImp(m, Wm)
#	print errComp_m
#	print np.sum(errComp_m)
#	print np.dot(m.T, m)/len(m)
#
#	print "--------------------------------------------------"
#	print "------------------Affine variant------------------"
#	print "--------------------------------------------------"
#	print "------------------Sphering------------------"
#	print "like before"
#
#	print "------------------Fitting------------------"
#	#Fitting:
#	zeta_c = calcZetacDataRefImp(z, p)
##	print zeta_c
#
##	z_p = z[p:]
#	W_c = calcRegressionCoeffAffineRefImp(z, p)
#	#Test fitting:
#	#print W
#	pre_c = np.dot(zeta_c, W_c.T) #np.dot(W, zeta.T).T
#	res_c = z_p-pre_c
#	print np.trace(np.dot(res_c.T, res_c))/len(res_c)
#	print empiricalRawErrorAffineRefImp(z, W_c)
#	errComp_c = empiricalRawErrorComponentsAffineRefImp(z, W_c)
#	print errComp_c
#	print np.sum(errComp_c)
#	print W_c
#
#	print "------------------Fitting Sinus------------------"
#	sig_x = lambda t: [np.sin(t)]
#	ts = range(0, 50)
#	sin_x = np.array([sig_x(t) for t in ts])
#	#print sin_x
#	sin_mean, sin_S, sin_z = calcSpheringParametersAndDataRefImp(sin_x)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
#	#Test Sphering:
#	#print sin_z
#	print "sin_z cov:"
#	print np.multiply(np.dot(sin_z.T, sin_z), 1.0/len(sin_z))
#
#	sin_p = 2
#	sin_zeta_c = calcZetacDataRefImp(sin_z, sin_p)
##	print zeta_c
#
#	sin_z_p = sin_z[sin_p:]
#	sin_W_c = calcRegressionCoeffAffineRefImp(sin_z, sin_p)
#	#Test fitting:
#	#print W
#	sin_pre_c = np.dot(sin_zeta_c, sin_W_c.T) #np.dot(W, zeta.T).T
#	sin_res_c = sin_z_p-sin_pre_c
#	print "sin_z, sin_W_c error:"
#	print np.trace(np.dot(sin_res_c.T, sin_res_c))/len(sin_res_c)
#	print empiricalRawErrorAffineRefImp(sin_z, sin_W_c)
#	sin_errComp_c = empiricalRawErrorComponentsAffineRefImp(sin_z, sin_W_c)
#	print sin_errComp_c
#	print np.sum(sin_errComp_c)
#	print "fitted sin W_c:"
#	print sin_W_c
#	print "analytic sin W_c:"
#	cs = np.cos(1.0)
#	sin_W_c_2 = np.array([[2*cs, -1.0, (-2.0+2.0*cs)*sin_S[0][0]*sin_mean[0]]])
#	print empiricalRawErrorAffineRefImp(sin_z, sin_W_c_2)
#	print sin_W_c_2
#
#	print "------------------PCA on Error Covariance------------------"
#	r_c = 1
#	#PCA on error covariance:
#	X_c = np.dot(res_c.T, res_c)
#	print X_c
#	Ar_c = calcExtractionForErrorCovRefImp(X_c, r_c)
#	m_c = np.dot(z, Ar_c)
#	Wm_c = calcRegressionCoeffAffineRefImp(m_c, p)
#	print m_c
#	print empiricalRawErrorAffineRefImp(m_c, Wm_c)
#	print np.dot(m_c.T, m_c)/len(m_c)
#
#	print "------------------PCA on Error Covariance - Noisy sine------------------"
#	sig_x2 = lambda t: [np.sin(t), np.random.rand(1)[0]]
#	#ts = range(0, 250)
#	nsin_x = np.array([sig_x2(t) for t in ts])
#	nsin_mean, nsin_S, nsin_z = calcSpheringParametersAndDataRefImp(nsin_x)#, threshold = 0.0000001, offset = 0, length = -1, besselsCorrection = 0)
#	#Test Sphering:
#	#print sin_z
#	print "nsin_z cov:"
#	print np.multiply(np.dot(nsin_z.T, nsin_z), 1.0/len(nsin_z))
#
#	nsin_p = 2
#	nsin_zeta_c = calcZetacDataRefImp(nsin_z, nsin_p)
###	print zeta_c
#
#	nsin_z_p = nsin_z[nsin_p:]
#	nsin_W_c = calcRegressionCoeffAffineRefImp(nsin_z, nsin_p)
#	#Test fitting:
#	#print W
#	nsin_pre_c = np.dot(nsin_zeta_c, nsin_W_c.T) #np.dot(W, zeta.T).T
#	nsin_res_c = nsin_z_p-nsin_pre_c
#	print "nsin_z, nsin_W_c error:"
#	print np.trace(np.dot(nsin_res_c.T, nsin_res_c))/len(nsin_res_c)
#	print empiricalRawErrorAffineRefImp(nsin_z, nsin_W_c)
#	nsin_errComp_c = empiricalRawErrorComponentsAffineRefImp(nsin_z, nsin_W_c)
#	print nsin_errComp_c
#	print np.sum(nsin_errComp_c)
##	print "fitted sin W_c:"
##	print nsin_W_c
##	print "analytic sin W_c:"
##	cs = np.cos(1.0)
##	nsin_W_c_2 = np.array([[2*cs, -1.0, (-2.0+2.0*cs)*nsin_S[0][0]*nsin_mean[0]]])
##	print empiricalRawErrorAffine(nsin_z, nsin_W_c_2)
##	print nsin_W_c_2
#
#	#print "------------------PCA on Error Covariance------------------"
#	nsin_r_c = 1
#	#PCA on error covariance:
#	nsin_X_c = np.dot(nsin_res_c.T, nsin_res_c)
#	print nsin_X_c
#	nsin_Ar_c = calcExtractionForErrorCovRefImp(nsin_X_c, nsin_r_c)
#	nsin_m_c = np.dot(nsin_z, nsin_Ar_c)
#	nsin_Wm_c = calcRegressionCoeffAffineRefImp(nsin_m_c, nsin_p)
#	#print nsin_m_c
#	print empiricalRawErrorAffineRefImp(nsin_m_c, nsin_Wm_c)
#	print np.dot(nsin_m_c.T, nsin_m_c)/len(nsin_m_c)
#	print "fitted nsin Wm_c:"
#	print nsin_Wm_c
#	print "analytic nsin Wm_c:"
#	SI = LA.inv(nsin_S.T)
#	factor = LA.norm(np.dot(np.array([1.0, 0]), SI))
#	Ar2 = np.multiply(np.dot(np.array([1.0, 0]), SI), 1.0/factor)
#	nsin_Wm_c_2 = np.array([[2*cs, -1.0, (-2.0+2.0*cs)*(1.0/factor)*nsin_mean[0]]])
#	print empiricalRawErrorAffineRefImp(nsin_m_c, nsin_Wm_c_2)
#	print nsin_Wm_c_2
#
	print ""
	print " _____________________________________________"
	print "/                                             \\"
	print "|---------------iteration stuff---------------|"
	print "\\_____________________________________________/"

	print ""
	#print zeta
	zeta0Ref = calcZeta0DataRefImp(zRef, p)
	#print zeta0
	ZZ0Ref = np.dot(zeta0Ref.T, zetaRef)
	zZRef = np.dot(z_pRef.T, zetaRef)
	ZZRef = np.dot(zetaRef.T, zetaRef)
	ZZIRef = invertByProjectionRefImp(ZZRef)
	W0Ref = np.dot(zZRef, ZZIRef)
	VRef = np.dot(ZZ0Ref, ZZIRef)
#	print W0Ref
#	print WRef
#	print ""
#	print VRef
#	print buildVRefImp(WRef)
#
#	print ""
#	print np.dot(ZZRef, ZZIRef)

	print "calc iterated error coeff:"
	k = 4
	X_kRef = calcErrorCoeffRefImp(zRef, WRef, k)
	Y_kRef = calcErrorCoeffConstLenRefImp(zRef, WRef, k)
	#print XRef
	print X_kRef
	#print Y_kRef


	X = np.zeros([len(WRef), len(WRef)])
	corList = [np.dot(zRef[p:].T, zRef[p:])]

	#for i in range(1, k+p+1):
	for i in range(1, p+1): #macht scheinbar eins zu viel)
		corList.append(np.dot(zRef[p:].T, zRef[p-i:-i]))

	zetaList = []
	lastLine = []#corList
	zetaBlockList = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]+np.outer(zRef[p-1], zRef[p-1-j])-np.outer(zRef[-1], zRef[-1-j]))
				else:
					#zetaLine.append(lastLine[j-1]-np.outer(zRef[-2], zRef[-2])+np.outer(zRef[p-2], zRef[p-2]))
					zetaLine.append(lastLine[j-1]+np.outer(zRef[p-1-i], zRef[p-1-j])-np.outer(zRef[-1-i], zRef[-1-j]))
					
		zetaList.append(zetaLine)
		lastLine = zetaLine

#	for i in range(p):
#		zetaBlockList.append(np.hstack(zetaList[i]))
#	ZZ0 = np.vstack(zetaBlockList)
#
##	for j in range(p):
##		lastLine[j] -= np.outer(zRef[-p-1], zRef[-2-j])
#	for i in range(p):
#		for j in range(p):
#			if j < i:
#				zetaList[i][j] = zetaList[j][i].T
#			else:
#				zetaList[i][j] -= np.outer(zRef[-2-i], zRef[-2-j])
#
#	#print np.hstack(lastLine)
#	for i in range(p):
#		zetaBlockList[i] = np.hstack(zetaList[i])
#	ZZ1 = np.vstack(zetaBlockList)
	#print ZZ1

#	for i in range(p):
#		for j in range(p):
#			if j < i:
#				zetaList[i][j] = zetaList[j][i].T
#			else:
#				zetaList[i][j] -= np.outer(zRef[-3-i], zRef[-3-j])
#
#	#print np.hstack(lastLine)
#	for i in range(p):
#		zetaBlockList[i] = np.hstack(zetaList[i])
#	ZZ2 = np.vstack(zetaBlockList)
#	print ZZ2

	#z_pk = data[p+k:]
	cov_p = np.dot(zRef[p:].T, zRef[p:])
	WV = WRef
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, VRef)
		zeta1_i = calcZetaDataRefImp(zRef, p, i)
#		print zRef
#		print zeta1_i
		#print len(zeta1_i)
		#X += np.dot((zRef[p+i:]-z_i_pre).T, zRef[p+i:]-z_i_pre)
#		X += np.dot(zRef[p+i:].T, zRef[p+i:]) - np.dot(z_i_pre.T, zRef[p+i:]) - np.dot(zRef[p+i:].T, z_i_pre) + np.dot(z_i_pre.T, z_i_pre)
		zZiRef = np.dot(zRef[p+i:].T, zeta1_i)
		if i > 0:
			for j in range(i+1, len(corList)):
				corList[j] -= np.outer(zRef[p+i-1], zRef[p-j+i-1])
			corList.append(np.dot(zRef[p+i:].T, zRef[:-p-i])) #ist scheinbar falsch
		zZi = np.hstack(corList[1+i:])#p+i+1])
#		print "+++"
#		print zZiRef
#		print zZi
#		print "==="
		ZZiRef = np.dot(zeta1_i.T, zeta1_i)

		#print np.hstack(lastLine)
		if i == 0:
			for l in range(p):
				zetaBlockList.append(np.hstack(zetaList[l]))
		else:
			for l in range(p):
				for j in range(p):
					if j < l:
						zetaList[l][j] = zetaList[j][l].T
					else:
						zetaList[l][j] -= np.outer(zRef[-(i+1)-l], zRef[-(i+1)-j])
			for l in range(p):
				zetaBlockList[l] = np.hstack(zetaList[l])
		ZZi = np.vstack(zetaBlockList)

#		print "i = "+str(i)
#		print zZiRef
#		print zZi
#		print "========="

		K = np.dot(zZi, WV.T)
		X += cov_p -K-K.T + np.dot(np.dot(WV, ZZi), WV.T)
		cov_p -= np.outer(zRef[p+i], zRef[p+i])

	print X
	print calcErrorCoeff(zRef, WRef, k)

	print "----------const len-----------"
	#Y = np.zeros([len(WRef), len(WRef)])
	z_pk = zRef[p+k:]
	WV = WRef
	Y = np.dot(z_pk.T, z_pk)*(k+1)
	corList = [np.dot(z_pk.T, z_pk)]
	for i in range(1, k+p+1):
		corList.append(np.dot(z_pk.T, zRef[p+k-i:-i]))

	zetaList = []
	lastLine = []#corList
	#zetaBlockList = []
	for i in range(p):
		zetaLine = []
		for j in range(p):
			if j < i:
				zetaLine.append(zetaList[j][i].T)
			else:
				if i == 0:
					zetaLine.append(corList[j]-np.outer(zRef[-1], zRef[-1-j])+np.outer(zRef[p+k-1], zRef[p+k-1-j]))
				else:
					#zetaLine.append(lastLine[j-1]-np.outer(zRef[-2], zRef[-2])+np.outer(zRef[p-2], zRef[p-2]))
					zetaLine.append(lastLine[j-1]-np.outer(zRef[-1-i], zRef[-1-j])+np.outer(zRef[p+k-1-i], zRef[p+k-1-j]))

		zetaList.append(zetaLine)
		lastLine = zetaLine

	for i in range(p):
		zetaList[i] = np.hstack(zetaList[i])
	ZZi = np.vstack(zetaList)
	pnl = len(WRef)*(p-1)
	for i in range(0, k+1):
		if i > 0:
			WV = np.dot(WV, VRef)
		zeta1_i = calcZetaDataRefImp(zRef, p, i)[k-i:]
		zZiRef = np.dot(z_pk.T, zeta1_i)
		zZi = np.hstack(corList[1+i:p+i+1])
		ZZiRef = np.dot(zeta1_i.T, zeta1_i)
		#ZZi = ZZ0
#		print "i = "+str(i)+":"
#		print ZZi
#		print ZZiRef
#		print "+++"+str(k)
		K = np.dot(zZi, WV.T)
		Y += -K-K.T + np.dot(np.dot(WV, ZZi), WV.T)
		if i < k:
			for j in range(p):
				lastLine[j] += -np.outer(zRef[-p-i-1], zRef[-j-2-i])+np.outer(zRef[k-1-i], zRef[p-i+k-j-2])
			line = np.hstack(lastLine)
			ZZi = np.vstack([np.hstack([ ZZi[pnl:].T[pnl:], line.T[:pnl]]), line])
#			print "===="
#			print np.hstack(lastLine)
	print Y_kRef
	print Y
	print calcErrorCoeffConstLen(zRef, WRef, k)
	print calcErrorCoeffConstLenRoll(zRef, WRef, k)
	
	print "-------------affine version---------------"
	print ""
	W_c = calcRegressionCoeffAffineRefImp(zRef, p)
	#print zeta
	zeta0_c = calcZeta0cDataRefImp(zRef, p)
#	print zeta0_c
#	print zeta_c
#	ZZ0_c = np.dot(zeta0_c.T, zeta_c)
#	zZ_c = np.dot(z_p.T, zeta_c)
#	ZZ_c = np.dot(zeta_c.T, zeta_c)
#	ZZI_c = invertByProjectionRefImp(ZZ_c)
#	W0_c = np.dot(zZ_c, ZZI_c)
	#The following idea of V_c is cumbersome, because it
	#puts error on the constant-one component of zeta_c,
	#if this reduces the overall error.
	#But the constant component may not be compromized in any case, so
	#this way to compute V_c is misleading:
#	V_c_cumb = np.dot(ZZ0_c, ZZI_c)
#	print V_c_cumb

#	print W0_c
	print W_c
	print ""
	#Better way:
	V_c = buildV_cRefImp(W_c)
	print V_c
	print "calc iterated error coeff:"
	X_k_c = calcErrorCoeffAffineRefImp(zRef, W_c, 4)
	print calcErrorCoeffAffine(zRef, W_c, 4)
	print X_k_c

	print "const len:"
	Y_k_c = calcErrorCoeffConstLenAffineRefImp(zRef, W_c, 3)
	print calcErrorCoeffConstLenAffine(zRef, W_c, 3)
	print Y_k_c
	print calcErrorCoeffConstLenAffineRoll(zRef, W_c, 3)

def insertRollTest():
	print "insert test"
	A = np.multiply(np.reshape(range(25), (5, 5)), 1.0)
	print A
#	B = np.array([[1.5, 2.5], [3.5, 4.5]])
#	print B
	print A[-3:-1, 3:5]
#	A[2:4, 3:5] = B
#	print A
#	print np.roll(np.roll(A, 2, 0), 2, 1)

if __name__ == "__main__":
	PFAReferenceImplementationTest()
	#compareIteration()
	#insertRollTest()
