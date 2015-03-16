from numpy import *
from os import listdir
import operator

def createDataSet():
	group = array([ [1.0,1.1] , [1.0,1.0], [0,0] , [0, 0.1]])
	labels = ['A' , 'A' , 'B' , 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize =dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDifMat = diffMat**2
	sqDistances = sqDifMat.sum(axis=1)
	# I am not taking square-root!!
	sortedDistances = sqDistances.argsort()
	classCount = {}
	for i in range(k):
		votedLabel = labels[sortedDistances[i]]
		classCount[votedLabel] = classCount.get(votedLabel,0) + 1
	sortedCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedCount[0][0]

def file2Matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append((listFromLine[-1]))
		index += 1
	return returnMat, classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest(k):
	hoRatio = 0.1
	datingDataMat, datingLabels = file2Matrix("datingTestSet.txt")
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], k)
		# print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i])
		if(classifierResult!=datingLabels[i]): errorCount += 1.0
	print "k value: %d, the total error rate is: %f" % (k,errorCount/float(numTestVecs))

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest(k):
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
		# print "the classifier came back with: %d, the real answer is %d" % (classifierResult, classNumStr)
		if (classifierResult != classNumStr):
			errorCount += 1.0
			print "mismatched digit: %s. Predicted:%d, Actual: %d" % (fileNameStr, classifierResult, classNumStr)
	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe total error rate is: %f" % (errorCount/float(mTest))
