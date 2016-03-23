import math
import random
import matplotlib.pyplot as plt

#Load data from a file into lists
def loadDataset(file,ignore):
	f=open(file,"r")
	lin=f.readline().split(',')
	data=[]
	while(lin[0]):
		ele=[]
		for i in xrange(0,len(lin)-1):
			if i not in ignore:
				ele.append(float(lin[i]))
		ele.append(lin[-1])
		data.append(ele)
		lin=f.readline().split(',')
	f.close()
	return data

#Divide the data randomly into 'folds' parts
def divideDataset(data,folds):
	random.shuffle(data)
	length=len(data)
	bound = int(length/folds)
	dataSet=[]
	for i in xrange(folds):
		dataSet.append(data[bound*i:bound*(i+1)])
	return dataSet

#Euclidean Distance Formula used
def euclideanDistance(point1, point2, length):
	distance=0
	for i in xrange(length):
		distance+=pow(point1[i]-point2[i],2)
	return math.sqrt(distance)

#Return k nearest neighbors of 'test' data
def nearestNeighbors(trainSet,test,k):
	neighbors=[]
	length=len(test)-1
	def getkey(item):
		return item[1]
	for i in xrange(len(trainSet)):
		neighbors.append( (trainSet[i],euclideanDistance(trainSet[i],test,length)) )
	neighbors.sort(key=getkey)
	k_neighbors=[]
	for i in neighbors[0:k]:
		k_neighbors.append(i[0])
	return k_neighbors

#Classify a test data into majority class of k neighbors.
#If a tie occurs, the nearest neighbor which is involved in tie is chosed and its class is used to classify the given test data.
def classify(neighbors):
	classes={}
	maxclass=[]
	maxx=-1
	for i in neighbors:
		clas = i[-1]
		if clas in classes:
			classes[clas] += 1
		else:
			classes[clas] = 1
		if classes[clas] > maxx:
			maxx=classes[clas]
			maxclass=[clas]
		elif classes[clas] == maxx:
			maxclass.append(clas)
	for i in neighbors:
		if maxclass[0]==i[-1]:
			return i[-1]

#Predicts class of all the test data in 'testSet' according to 'trainSet'
def predict(trainSet,testSet,k):
	predictions=[]
	for i in xrange(len(testSet)):
		neighbors = nearestNeighbors(trainSet,testSet[i],k)
		predictions.append(classify(neighbors))
	return predictions

#Calculates Accuracy i.e, number of correct predictions by total predictions multiplied by 100
def calcAccuracy(testSet, predicted):
	correct=0
	length=len(testSet)
	for i in xrange(length):
		if testSet[i][-1]==predicted[i]:
			correct+=1
	return (correct/float(length)) * 100.0

#Plotting graph of Mean Accuracy with K values
def plot(K, plotData,plotData2):
	cols=["red","darkgreen","blue","orange"]
	for i in xrange(4):
		plt.plot(K,plotData[i],label=("Fold #"+str(i+2)),color=cols[i])
		plt.errorbar(K,plotData[i],yerr=plotData2[i],color=cols[i])
	plt.legend()
	plt.xlabel("K",fontsize=15)
	plt.ylabel("Mean Accuracyy",fontsize=15)
	plt.suptitle("Ionosphere Data Set",fontsize=20)
	plt.xticks(K)
	plt.show()

def main():
	plotData=[]
	plotData2 = []
	data=loadDataset('ionosphere.data',[])
	#For each fold in [2,5] and for each k in [1,5], accuracy is measured
	for folds in xrange(2,6):
		dataSet = divideDataset(data,folds)
		plotFold=[]
		plotFold2 = []
		for k in xrange(1,6):
			averageAccuracy=0.0
			accuracies=[]
			for i in xrange(folds):
				testSet=dataSet[i]
				trainSet=[]
				for j in xrange(folds):
					if j!=i:
						trainSet+=dataSet[j]
				predictions=predict(trainSet,testSet,k)
				accuracies.append(calcAccuracy(testSet,predictions))
				averageAccuracy+=accuracies[-1]
			averageAccuracy/=float(folds)
			dev=0.0
			for i in accuracies:
				dev+=pow((i-averageAccuracy),2)
			dev/=float(folds)
			dev=math.sqrt(dev)
			plotFold.append(averageAccuracy)
			plotFold2.append(dev)
		plotData.append(plotFold)
		plotData2.append(plotFold2)
	K=[1,2,3,4,5]
	plot(K,plotData,plotData2)
	
main()