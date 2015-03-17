from numpy import *

def loadDataSet():
	dataMat = []
	labelMat = []
	fr=open('testSet.txt')
	for line in fr.readlines():
		lineArr = line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def sigmoid(inx):
	return 1.0/(1+exp(-inx))

def gradAscent(dataMatIn, classLabels, maxCycles = 20):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	alpha = 0.001
	weights = ones((n,1))
	for k in range(maxCycles):
		z = dataMatrix*weights
		h = sigmoid(z)
		error = (labelMat - h)
		print 'Error %d : %.4f' % (k,(error.transpose()*error)[0][0])
		weights = weights + alpha*dataMatrix.transpose()*error
	return weights

def predict(weights,dataMatIn, classLabels):
	dataMatrix = mat(dataMatIn)
	labelMat = mat(classLabels).transpose()
	m, n = shape(dataMatrix)
	predict = sigmoid(dataMatrix*weights)
	errorcount = 0
	for i in range(m):
		prediction = 0
		if predict[i]>=0.5: prediction = 1
		result = ''
		if prediction!=labelMat[i]: 
			result = ' that is wrong!!!'
			errorcount += 1
		print 'predict: %d, actual: %d %s' % (prediction, labelMat[i], result)
	errorrate = float(errorcount)/m
	print 'Overall errorrate: %.4f' % errorrate

def plotBestFit(wei):
	import matplotlib.pyplot as plt 
	weights = wei #.getA()
	dataMat, labelMat = loadDataSet()
	dataArr = array(dataMat)
	m = shape(dataArr)[0]
	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []
	for i in range(m):
		if int(labelMat[i])==1:
			xcord1.append(dataArr[i,1])
			ycord1.append(dataArr[i,2])
		else:
			xcord2.append(dataArr[i,1])
			ycord2.append(dataArr[i,2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
	ax.scatter(xcord2, ycord2, s=30, c='green')
	x = arange(-3.0,3.0, 0.1)
	y = (-weights[0]-weights[1]*x)/weights[2]
	ax.plot(x,y)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()

def plotAfterIterations(n):
	dataMat, labelMat = loadDataSet()
	weights = gradAscent(dataMat, labelMat, n)
	plotBestFit(weights)

def stocGradAscent0(dataMatrix, classLabels, cycles = 10):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for k in range(cycles):
	    for i in range(m):
	        h = sigmoid(sum(dataMatrix[i]*weights))
	        error = classLabels[i] - h
	        weights = weights + alpha * error * array(dataMatrix[i])
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
  	m,n = shape(dataMatrix)
  	weights=ones(n)
  	for j in range(numIter):
  		dataIndex = range(m)
  		for i in range(m):
  			alpha=4/(1.0+j+i)+0.01
  			randIndex = int(random.uniform(0,len(dataIndex)))
  			h = sigmoid(sum(dataMatrix[randIndex]*weights))
  			error = classLabels[randIndex] - h
  			weights = weights + alpha * error * array(dataMatrix[randIndex])
  			del(dataIndex[randIndex])
  	return weights
