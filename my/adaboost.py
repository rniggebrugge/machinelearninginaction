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

def buildStump(dataArr, classLabels, D):
	dataMatrix = mat(dataArr);
	labelMat = mat(classLabels).T 
	m,n = shape(dataMatrix)
	numSteps = 10.0
	bestStump = {}
	bestClasEst = mat(zeros((m,1)))
	minError = inf 
	for i in range(n):
		rangeMin = dataMatrix[:,i].min()
		rangeMAx = dataMatrix[:,i].max()
		stepSize = (rangeMAx - rangeMin)/numSteps
		for j in range(-1, int(numSteps)+1):
			threshVal = (rangeMin + float(j) * stepSize)
			for inequal in ['lt', 'gt']:
				predictedVal = stumpClassify(dataMatrix)


def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
	weakClassArr = []
	labelMat = mat(classLabels).T 
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)
	aggClasEst = mat(zeros((m,9)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D)
		# print "D:", D.T
		alpha = float(log((1.0-error)/max(error, 1e-16)))+log(8)
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)

		errArr = -1* mat(ones((m,1)))
		errArr[classEst==labelMat] = 1

		# print errArr.T
		expon = -1*alpha*errArr
		D = multiply(D, exp(expon))
		D = D/D.sum()
		# print D.T
		# for j in range(m):
		# 	aggClasEst[j,int(classEst[j])]+=alpha;
		# aggClasEst += alpha*classEst
		# aggClasEst = (i*aggClasEst + classEst)/(i+1)
		# print "aggClasEst: \n", aggClasEst.T
		# aggErrors = multiply(sign(aggClasEst) != mat(classLabels).T, ones((m,1)) )
		# errorRate = aggErrors.sum()/m
		# print "%d / total error: %.3f" % (i+1, errorRate)
		# if errorRate==0.0: break
		# raw_input()
	# return weakClassArr, aggClasEst

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

def loadKaggleSet(filename):
	numFeat = len(open(filename).readline().split(','))
	dataMat = []
	labelMat =[]
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split(',')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat, labelMat