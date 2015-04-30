from numpy import *

def loadSimpData():
	datMat = matrix([[1., 2.1],
		[2., 1.1],
		[1.3, 1. ],
		[1. , 1.],
		[2., 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat, classLabels

def loadSimpDataMulti():
	datMat = matrix([[1., 2.1001],
		[2., 1.1002],
		[1.3, 1.0003 ],
		[1. , 1.0004],
		[2., 1.0005],
		[3., 2.0006],
		[6., 0.0007],
		[2., 10.0008],
		[6., 8.0009],
		[19., 11.00010],
		[8., 4.00011],
		[7.6, 5.00012],
		[5., 1.20013]])
	classLabels = [1.0, 1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 4.0, 4.0, 4.0, 3.0, 2.0, 3.0]
	return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def buildSplitMatrix(dataArr, numSteps = 10.0):
	dataMatrix = mat(dataArr);
	m,n = shape(dataMatrix)
	splitMatrix = zeros((m, (int(numSteps)+2)*n))
	threshVals = zeros((n, int(numSteps)+2))
	column = 0
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMAx = dataMatrix[:,i].max()
		stepSize = (rangeMAx - rangeMin)/numSteps
		for j in range(-1, int(numSteps)+1):
			threshVal = (rangeMin + float(j) * stepSize)
			threshVals[i,j+1] = threshVal
			predictedValsLT = stumpClassify(dataMatrix, i, threshVal, 'lt')
			splitMatrix[:, column] = predictedValsLT[:,0]
			column += 1
	return splitMatrix, threshVals

def buildStump(dataArr, classLabels, D, splitMatrix):
	dataMatrix = mat(dataArr);
	labelMat = mat(classLabels).T 
	m,n = shape(dataMatrix)
	bestStump = {}
	bestClassEst = mat(zeros((m,1)))
	minError = inf 
	numSteps = shape(splitMatrix)[1]/n - 2.
	readColumn = 0

	for i in range(n):
		for j in range(-1, int(numSteps)+1):
			for inequal in [1, -1]:  ##!!!!!! 1 correspondts with lt, -1 with gt
				predictedVals = inequal * mat(splitMatrix[:,readColumn])
				errArr =mat(ones((m,1)))
				errArr[predictedVals.T == labelMat] =0
				weightedError = D.T*errArr
#				print "split: dim %d, thresh %.2f, inequal %s, error %.3f" % (i, threshVal, inequal, weightedError)
				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i 
					bestStump['thresh'] = j
					bestStump['ineq'] = 'lt' if inequal==1 else 'gt'
			readColumn += 1
	
	return bestStump, minError, bestClassEst



def adaBoostTrainDS(dataArr, classLabels, numIt = 40, splits = 5, multi = False):
	weakClassArr = []
	classMat = mat(classLabels)
	nClassifications = 1

	m = shape(dataArr)[0]
	if multi:
		uniqueLabels = unique(array(classLabels))
		nClassifications = size(uniqueLabels)
		multiLabelMatrix = mat(zeros((m, nClassifications)))
		column = 0
		for i in uniqueLabels:
			multiLabelMatrix[:,column] = mat((classMat ==i)*2-1).T
			column += 1


	splitMatrix, threshVals = buildSplitMatrix(dataArr, float(splits))

	for classification in range(nClassifications):

	#	print "\n************************************************************************"
		print "Performing classification %d " % (classification+1)
#		print "************************************************************************"

		D = mat(ones((m,1))/m)
		aggClassEst = mat(zeros((m,1)))
		if multi:
			ul = uniqueLabels[classification]
#			print "   ---  Looking for %d ---" % ul
			classLabels_revised = mat((classMat == ul)*2-1)
		else:
			classLabels_revised = mat(classLabels)

		for i in range(numIt):
			bestStump, error, classEst = buildStump(dataArr, classLabels_revised,D, splitMatrix)
			alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
			bestStump['alpha'] = alpha
			weakClassArr.append(bestStump)
			expon = multiply(-1*alpha*mat(classLabels_revised).T, classEst.T)
			D = multiply(D, exp(expon))
			D = D/D.sum()
			aggClassEst += alpha*classEst.T
	#		print "aggClassEst: ", aggClassEst.T
			aggErrors = multiply(sign(aggClassEst) != classLabels_revised.T, ones((m,1)))
			errorRate = aggErrors.sum()/m
			print "class %d , iteration: %d,  error rate: %.3f " % ( int(ul) , i , errorRate)
	#		tp = sum((aggClassEst>0) & (classMat==1))
	#		fp = sum((aggClassEst>0) & (classMat==-1))
	#		fn = sum((aggClassEst<=0) & (classMat==1))
	#		precision = float(tp)/(tp+fp)
	#		recall = float(tp)/(tp+fn)
	#		print "total error: %.3f, precision: %.3f, recall: %.3f " % \
	#			(errorRate, precision, recall)
	#		print "total error: %.3f " % errorRate
	#		if errorRate == 0.0: break

		print "Final result for class %d" % ul
#		print "\nP: " , mat(aggClassEst.T>0)*1
#		print "\nA: " , mat(classLabels_revised>0)*1
		tp = (mat((aggClassEst>0) & (classLabels_revised.T>0))*1).sum()
		fp = (mat((aggClassEst>0) & (classLabels_revised.T<0))*1).sum()
		fn = (mat((aggClassEst<=0) & (classLabels_revised.T>0))*1).sum()
		precision = float(tp)/(tp+fp)
		recall = float(tp)/(tp+fn)
		f1 = 2*float(precision)*recall/(precision+recall)

		print "-----------------------------------------------------"
		print "TP: ", tp
		print "FP: ", fp
		print "FN: ", fn
		print "Precision: %.2f " % precision
		print "Recall: %.2f" % recall
		print "F1-score: %.2f" % f1

#		print "\n" , aggClassEst.T
	return weakClassArr


def adaClassify(datToClass, classifierArr):
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
			classifierArr[i]['thresh'], \
			classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		print aggClassEst
	return sign(aggClassEst)

def plotROC(predStrengts, classLabels):
	import matplotlib.pyplot as plt 
	cur = (1.0, 1.0)
	ySum = 0.0
	numPosClas = sum(array(classLabels)==1.0)
	yStep = 1/float(numPosClas)
	xStep = 1/float(len(classLabels)-numPosClas)
	sortedIndices = predStrengts.argsort()
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndices.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0
			delY = yStep
		else:
			delX = xStep
			delY = 0
			ySum += cur[1]
		ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
		cur = (cur[0]-delX, cur[1]-delY)	
	ax.plot([0,1],[0,1], 'b--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for AdaBoost Horse Colic Detection System')
	ax.axis([0,1,0,1])
	plt.show()
	print "The area under the curve is: ", ySum*xStep

def loadData(filename, splitChar = '\t', lookFor = 0):
	numFeat = len(open(filename).readline().split(splitChar))
	dataMat = []
	labelMat =[]
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split(splitChar)
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	labelMat = mat(labelMat)
	if lookFor!=0:
		print "relabelling"
		labelMat[labelMat!=lookFor] = -1
		labelMat[labelMat==lookFor] = 1
	
	# dataMat = log(mat(dataMat)+1)
	return dataMat, labelMat