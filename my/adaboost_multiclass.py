from numpy import *

def assignIntervals(dataMatrix, dimen, intervals, assignBits):
	intsMat = mat(intervals)
	retArray = ones((shape(dataMatrix)[0],1))
	m = shape(intsMat)[1]
	lowerBound = intsMat[0]
	for i in range(1,m):
		upperBound = intsMat[0,i]
		doAssign = assignBits & 1
		if doAssign==0:
			retArray[(dataMatrix[:,dimen] < upperBound) & (dataMatrix[:,dimen] >=lowerBound)] = -1.0
		lowerBound = upperBound
		assignBits = assignBits >>1
	doAssign = assignBits & 1
	if doAssign==0:
		retArray[dataMatrix[:,dimen] >=lowerBound] = -1.0
	return retArray

def buildSplit(dataArr, classLabels, D):
	dataMatrix = mat(dataArr);
	labelMat = mat(classLabels).T 
	nLabels = int(labelMat.max())
	m,n = shape(dataMatrix)
	numSteps = 7.0
	permutations = 2**int(numSteps)

	# could also be one less, to avoid ALL bits to be one.
	# in the loop later I will start at 1, avoiding ALL bits to be zero
	# as this would favour a configuration with all parts having NOT class 1 (or other rare class)

	bestSplit = {}
	bestClassEst = mat(zeros((m,1)))
	bestClass = 0
	minError = inf 
	for i in range(n):
		print 'Considering feature: %d' % i
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax - rangeMin)/numSteps
		intervals = [round(float(x)*stepSize + rangeMin,1) for x in range(int(numSteps))]
		intervals[0] -= 0.1;
		# print intervals, rangeMin, rangeMax
		for c in range(nLabels):
			assignClass = c+1
			for assignBits in range(1, permutations):
				predictedVals = assignIntervals(dataMatrix, i, intervals, assignBits)
				errArr = mat(zeros((m,1)))
				errArr[(predictedVals == 1) & (labelMat != assignClass)] = 1
				errArr[(predictedVals == -1) & (labelMat == assignClass)] =1
				weightedError = D.T * errArr
				# print "split: dim %d, class %d, split %s, the weighted error is %.3f" %\
				# 	(i, assignClass, bin(assignBits), weightedError)
				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestClass = assignClass
					bestSplit['dim'] = i 
					bestSplit['intervals'] = intervals
					bestSplit['class'] = assignClass
					bestSplit['bits'] = assignBits
	print "best split: dim %d, class %d, split %s, the weighted error is %.3f" %\
		(bestSplit['dim'], bestSplit['class'], bin(bestSplit['bits']), minError)				
	return bestSplit, minError, bestClassEst, bestClass


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