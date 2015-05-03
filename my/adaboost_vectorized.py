from numpy import *

"""

"""


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def quantilesMatrix(datArr, numSteps = 10):
	ns = int(numSteps)
	mtrx = mat(datArr).astype(float)
	minV = mtrx.min(0)
	maxV = mtrx.max(0)
	stepSize = maxV/ns
	lowerBound = -stepSize
	for i in range(0, ns+2):
		upperBound = lowerBound + stepSize
		mtrx[(mtrx>lowerBound) & (mtrx<=upperBound)] = -(i+1)
		lowerBound = upperBound
	mtrx = -mtrx
	return mtrx

def buildStumpQuantiles(classLabels, D, splitMatrix, numSteps):
	ns = int(numSteps)
	labelMat = mat(classLabels)
	bestStump = {}
	minErr = inf
	for i in range(1,ns+2):
		for j in range(i, ns+2):
			w = -splitMatrix.copy()
			w[(w<=-i) &(w>=-j)]=1
			w[w!=1]=-1
			# print w[0:4,0:4]

			Err = -multiply(labelMat, w.T)
			Err[Err==-1] = 0
			# print "error: ",sum(Err)
			# print labelMat
			# print Err[0:8,0:8]
			weights = Err * D
			localMinError = weights.min()
			# print localMinError
			if localMinError<minErr:
				minErr = localMinError
				quantily = i
				bestStump['dim'] = weights.argmin()
				bestClassEst = w[:, weights.argmin()]
 	return bestStump, minErr, bestClassEst

def buildSplitMatrix(dataArr, numSteps = 10):
	numSteps = int(numSteps)
	mtrx = mat(dataArr).astype(float);
	rangeMin = mtrx.min(0)
	rangeMax = mtrx.max(0)
	stepSize  = (rangeMax - rangeMin)/numSteps
	for i in range(-1, numSteps+1):
		temp = mtrx - rangeMin - i*stepSize
		temp[temp==0] = -1e-16
		temp = sign(temp)
		if i ==-1:
			retMatrix = temp
		else:
			retMatrix = concatenate((retMatrix, temp), axis=1)
	return retMatrix

def buildStump(dataArr, classLabels, D, splitMatrix):
	dataMatrix = mat(dataArr)
	nFeatures = shape(dataMatrix)[1]
	labelMat = mat(classLabels)
	bestStump = {}
	ltgt_mtrx = concatenate((splitMatrix, -splitMatrix), 1)
	Err = -multiply(labelMat.T, ltgt_mtrx)	
	Err[Err==-1] = 0
	weights = D.T*Err
	minError = weights.min()
	split = weights.argmin()
	# print "Split %d , feature %d, threshhold %d " %(split,mod(split, nFeatures) +1, split/2)
	bestClassEst = ltgt_mtrx[:,split]
	bestStump['split'] = split 
 	return bestStump, minError, bestClassEst

def logicalComparison(v, m, sv, cv, operator):
	if sv=='inverse':
		v = -v
	if cv=='inverse':
		m = -m
	combi = m + v
	if operator == 'AND':
		combi = (combi==2)*2-1
	elif operator =='OR':
		combi = (combi>-2)*2-1
	elif operator == 'XOR':
		combi = (combi==0)*2-1
	return combi

def buildStumpComplex(dataArr, classLabels, D, splitMatrix):
	dataMatrix = mat(dataArr)
	nFeatures = shape(dataMatrix)[1]
	labelMat = mat(classLabels)
	bestStump = {}
	nColumns = shape(splitMatrix)[1]
	minError = inf

	for i in range(nColumns-1):
		vector = splitMatrix[:,i]
		compareMatrix = splitMatrix[:,i+1:]
		for sv in ['normal', 'inverse']:
			for sm in ['normal', 'inverse']:
				for operator in ['AND', 'XOR', 'OR']:
					combi = logicalComparison(vector, compareMatrix, sv, sm, operator)
					Err = -multiply(labelMat.T, combi)
					Err[Err==-1] = 0
					weights = D.T*Err
					localMinError = weights.min()
					if localMinError < minError:
						minError = localMinError
						split = weights.argmin()
						bestStump['feature_1'] = i
						bestStump['feature_2'] = i + 1 + split
						bestStump['orientation_1'] = sv
						bestStump['orientation_2'] = sm
						bestStump['operator'] = operator
						bestClassEst = combi[:, split]
 	return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40, splits = 5, multi = False):
	weakClassArr = []
	datMat = mat(dataArr)
	labelMat = mat(classLabels)
	nClassifications = 1
	m = shape(datMat)[0]

	if multi:
		uniqueLabels = unique(array(classLabels))
		nClassifications = size(uniqueLabels)
		multiLabelMatrix = mat(zeros((m, nClassifications)))
		column = 0
		for i in uniqueLabels:
			multiLabelMatrix[:,column] = mat((labelMat ==i)*2-1).T
			column += 1


	# splitMatrix = buildSplitMatrix(datMat, splits)
	splitMatrix = quantilesMatrix(datMat, splits)

	for classification in range(nClassifications):

		# print "\n************************************************************************"
		print "Performing classification %d " % (classification+1)
		# print "************************************************************************"

		D = mat(ones((m,1))/m)
		aggClassEst = mat(zeros((m,1)))
		if multi:
			ul = uniqueLabels[classification]
			print "   ---  Looking for %d ---" % ul
			classLabels_revised = mat((labelMat == ul)*2-1)
		else:
			ul = 1
			classLabels_revised = mat(classLabels)

		for i in range(numIt):
			# bestStump, error, classEst = buildStump(datMat, classLabels_revised,D, splitMatrix)
			# bestStump, error, classEst = buildStumpComplex(datMat, classLabels_revised,D, splitMatrix)
			bestStump, error, classEst = buildStumpQuantiles(classLabels_revised, D, splitMatrix, splits)
			alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
			bestStump['alpha'] = alpha
			weakClassArr.append(bestStump)
			expon = multiply(-1*alpha*mat(classLabels_revised).T, classEst)
			D = multiply(D, exp(expon))
			D = D/D.sum()
			aggClassEst += alpha*classEst
			aggErrors = multiply(sign(aggClassEst) != classLabels_revised.T, ones((m,1)))
			errorRate = aggErrors.sum()/m
			print "class %d , iteration: %d,  error rate: %.3f " % ( int(ul) , i , errorRate)
			if errorRate == 0.0: 
				print "Zero error. Break out."
				break

		print "\n================================================================\ntotal iterations: ",i		
		tp = (mat((aggClassEst>0) & (classLabels_revised.T>0))*1).sum()
		fp = (mat((aggClassEst>0) & (classLabels_revised.T<0))*1).sum()
		fn = (mat((aggClassEst<=0) & (classLabels_revised.T>0))*1).sum()
		precision = float(tp)/(1e-16+tp+fp)
		recall = float(tp)/(1e-16+tp+fn)
		f1 = 2*float(precision)*recall/(1e-16+precision+recall)

		print "Final result for class %d" % ul
		print "TP: ", tp
		print "FP: ", fp
		print "FN: ", fn
		print "Precision: %.2f " % precision
		print "Recall: %.2f" % recall
		print "F1-score: %.2f" % f1
		print "-----------------------------------------------------"

#		print "\n" , aggClassEst.T
	return weakClassArr


