from numpy import *

def loadSimpData():
	datMat = matrix([[1., 2.1],
		[2., 1.1],
		[1.3, 1. ],
		[1. , 1.],
		[2., 1.]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def buildSplitMatrix(dataArr):
	dataMatrix = mat(dataArr);
	m,n = shape(dataMatrix)
	numSteps = 20.0
	splitMatrix = zeros((m, (int(numSteps)+2)*n))
	threshVals = zeros((n, int(numSteps)+2))
	column = 0
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMAx = dataMatrix[:,i].max()
		stepSize = (rangeMAx - rangeMin)/numSteps
		for j in range(-1, int(numSteps)+1):
			threshVal = (rangeMin + float(j) * stepSize)
			threshVals[i,j] = threshVal
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
			for inequal in [1, -1]:
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
					bestStump['ineq'] = inequal
			readColumn += 1
	return bestStump, minError, bestClassEst



def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)
	splitMatrix, threshVals = buildSplitMatrix(dataArr)
	print threshVals
	aggClassEst = mat(zeros((m,1)))

	for i in range(numIt):
		print "===================",i, "========================================"
		bestStump, error, classEst = buildStump(dataArr, classLabels,D, splitMatrix)
		alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		expon = multiply(-1*alpha*mat(classLabels).T, classEst.T)
		D = multiply(D, exp(expon))
		D = D/D.sum()
		aggClassEst += alpha*classEst.T
#		print "aggClassEst: ", aggClassEst.T
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
		errorRate = aggErrors.sum()/m
		print "total error: ", errorRate
#		if errorRate == 0.0: break
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

def loadData(filename, splitChar = '\t'):
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
	return dataMat, labelMat