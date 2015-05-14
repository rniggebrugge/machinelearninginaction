from numpy import *

"""

"""

def buildSplitMatrix(dataArr, numSteps = 10):
	numSteps = int(numSteps)
	mtrx = mat(dataArr).astype(float);
	nFeatures = shape(mtrx)[1]
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
	halfsize = shape(retMatrix)[1]
	retMatrix2 = concatenate((retMatrix, -retMatrix), 1);
	for i in range(nFeatures-2):
		lowerLimit1 = i*(numSteps+2)
		upperLimit1 = lowerLimit1+numSteps+1
		for j in range(3):
			for k in range(2):
				if (j>0 or k>0):
					lowerLimit2 = lowerLimit1 + j*(numSteps+2) + k * halfsize
					upperLimit2 = lowerLimit2 + numSteps + 1
					addedBlock = retMatrix2[:,lowerLimit1:upperLimit1] + retMatrix2[:,lowerLimit2:upperLimit2]
					addedBlock = (addedBlock==2)*2-1
					print addedBlock.max()
					retMatrix2 = concatenate((retMatrix2, addedBlock),1)
					print lowerLimit1, upperLimit1, lowerLimit2, upperLimit2
	print shape(retMatrix2)		
	return retMatrix2

def getErrorMatrix(splitMatrix, classLabels):
	labelMat = mat(classLabels)
	Err = -multiply(labelMat.T, splitMatrix)	
	Err[Err==-1] = 0	
	return Err

def buildStump(D, splitMatrix, Err):
	bestStump = {}
	weights = D.T*Err
	minError = weights.min()
	split = weights.argmin()
	bestClassEst = splitMatrix[:,split]
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

		Err = getErrorMatrix(splitMatrix, classLabels_revised)

		for i in range(numIt):
			bestStump, error, classEst = buildStump(D, splitMatrix, Err)

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


