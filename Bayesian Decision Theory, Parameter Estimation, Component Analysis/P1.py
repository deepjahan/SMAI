import random
from math import log

def loadData(file):
	f=open(file,"r")
	a=f.readline()
	a=f.readline()
	data=[]
	while(a):
		data.append(a.split(";"))
		a=f.readline()
	return data


class naiveBayes(object):
	def __init__(self, train, test, ignore):
		self.train = train
		self.test = test
		self.ignore=ignore
	
	def setTrain(self,train):
		self.train=train
	
	def setTest(self,test):
		self.test=test

	def trainData(self):
		self.prior={}
		self.likelihood=[]
		d=len(self.train[0])-1
		for i in range(d):
			self.likelihood.append({})
		for i in self.train:
			if i[-1] not in self.prior.keys():
				self.prior[i[-1]]=1
			else:
				self.prior[i[-1]]+=1
		for i in self.train:
			for j in range(d):
				if i[j] not in self.likelihood[j].keys():
					tobeapp={}
					for k in self.prior.keys():
						if(i[-1]==k):
							tobeapp[k]=1.0
						else:
							tobeapp[k]=0.0
					self.likelihood[j][i[j]]=tobeapp
				else:
					self.likelihood[j][i[j]][i[-1]]+=1.0
		for i in self.prior.keys():
			for j in range(d):
				for k in self.likelihood[j].keys():
					if self.likelihood[j][k][i]==0.0:
						self.likelihood[j][k][i]="Nil"
					else:
						self.likelihood[j][k][i]=log(float(self.likelihood[j][k][i]))-log(float(self.prior[i]))
		lenTrain=len(self.train)
		for i in self.prior.keys():
			self.prior[i]=log(float(self.prior[i])) - log(float(lenTrain))

	def testData(self):
		accuracy=0
		for i in self.test:
			finalClass=self.prior.keys()[0]
			postMax=-10000000000
			for j in self.prior.keys():
				post=self.prior[j]
				for k in range(len(i)-1):
					if k not in self.ignore:
						if i[k] in self.likelihood[k].keys() and self.likelihood[k][i[k]][j]!="Nil" and i[k]!='"unknown"':
							post+=self.likelihood[k][i[k]][j]
				if(post>postMax):
					postMax=post
					finalClass=j
			if finalClass==i[-1]:
				accuracy+=1
		accuracy=float(accuracy)/float(len(self.test))
		return accuracy*100

def main():
	data=loadData("./bank/bank.csv")
	random.shuffle(data)
	length=len(data)
	trainData=data[0:length/2]
	testData=data[length/2:length]
	ignore=[5,11,15]
	B=naiveBayes(trainData,testData,ignore)
	B.trainData()
	print B.testData(),"%"
	for i in range(9):
		random.shuffle(data)
		trainData=data[0:length/2]
		testData=data[length/2:length]
		B.setTest(testData)
		B.setTrain(trainData)
		B.trainData()
		print B.testData(),"%"

main()