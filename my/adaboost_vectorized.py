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


	splitMatrix = buildSplitMatrix(datMat, splits)

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
			bestStump, error, classEst = buildStump(datMat, classLabels_revised,D, splitMatrix)
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


