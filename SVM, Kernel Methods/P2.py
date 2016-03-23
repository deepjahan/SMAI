import numpy as np
import math
from sklearn import svm

def loadData(file):
	data=np.loadtxt(file)
	return data

def getClassIndex(file):
	classes={}
	data=np.loadtxt(file,unpack=True,dtype=int)
	for i in range(len(data)):
		if data[i] not in classes.keys():
			classes[data[i]]=[i]
		else:
			classes[data[i]].append(i)
	return classes

def Linear(x,y,dummy):
	return np.dot(x,y)

def RBF(x,y,sig_sq):
	return math.exp(-np.dot(x-y,x-y)/(2*sig_sq))

def calcOptimalSig_sq(data,classes):
	keys=classes.keys()
	W_den=0.0
	for i in keys:
		W_den+=len(classes[i])*len(classes[i])
	W=0.0
	for i in keys:
		for t in range(len(classes[i])):
			for k in range(len(classes[i])):
				W+=np.dot(data[classes[i][t]]-data[classes[i][k]],data[classes[i][t]]-data[classes[i][k]])
	W/=W_den
	B_den=0.0
	for i in keys:
		for j in keys:
			if i!=j:
				B_den+=len(classes[i])*len(classes[j])
	B=0.0
	for i in keys:
		for j in keys:
			if i!=j:
				for t in range(len(classes[i])):
					for k in range(len(classes[j])):
						B+=np.dot(data[classes[i][t]]-data[classes[j][k]],data[classes[i][t]]-data[classes[j][k]])
	B/=B_den
	return (B-W)/(4.0*math.log(B/W))

def getEigenVectors(mat,k):
	val,vec = np.linalg.eigh(mat)
	eig=zip(val,vec.T)
	eig.sort(reverse=True)
	return zip(*eig)[1][:k]

def getKernelMatrix(data1,n1,data2,n2,kernel,sig_sq):
		kern=[]
		for i in range(n1):
			kern_i=[]
			for j in range(n2):
				kern_i.append(kernel(data1[i],data2[j],sig_sq))
			kern.append(np.array(kern_i))
		return np.array(kern)

class KPCA(object):
	def __init__(self, act, classes, kernel):
		self.act = act
		self.N = len(act)
		self.kernel=kernel
		self.classes=classes
		self.sig_sq=calcOptimalSig_sq(self.act,self.classes)
		self.mean = self.act.mean(0)
		self.act=self.act-self.mean
	def generateTrain(self,k):
		K=getKernelMatrix(self.act,self.N,self.act,self.N,self.kernel,self.sig_sq)
		eig=getEigenVectors(K,k)
		self.eig=np.array(eig)
		return np.dot(K,self.eig.T)
	def generateTest(self,test):
		test=test-self.mean
		return np.dot(getKernelMatrix(test,len(test),self.act,self.N,self.kernel,self.sig_sq),self.eig.T)

class KLDA(object):
	def __init__(self, act, classes, kernel):
		self.act = act
		self.N = len(act)
		self.kernel=kernel
		self.classes=classes
		self.sig_sq=calcOptimalSig_sq(self.act,self.classes)
		self.mean = self.act.mean(0)
		self.act=self.act-self.mean
	def generateTrain(self):
		self.K={}
		for i in self.classes.keys():
			data=[]
			for j in self.classes[i]:
				data.append(self.act[j])
			data=np.array(data)
			self.K[i]=getKernelMatrix(self.act,self.N,data,len(data),self.kernel,self.sig_sq)
		self.N_mat=np.zeros((self.N,self.N))
		for i in self.classes.keys():
			l=len(self.classes[i])
			self.N_mat+=np.dot(np.dot(self.K[i],(np.identity(l)-np.ones((l,l))/l)),self.K[i].T)
		self.M_mat=[]
		for i in self.classes.keys():
			M=[]
			for j in range(self.N):
				M.append(np.sum(self.K[i]))
			self.M_mat.append(np.array(M)/len(self.classes[i]))
		self.M_mat=np.array(self.M_mat)
		self.alpha= np.dot(np.linalg.inv(self.N_mat),self.M_mat[1]-self.M_mat[0])
		return np.dot(self.alpha,getKernelMatrix(self.act,self.N,self.act,self.N,self.kernel,self.sig_sq))
	def generateTest(self,test):
		test=test-self.mean
		return np.dot(self.alpha,getKernelMatrix(self.act,self.N,test,len(test),self.kernel,self.sig_sq))

def checkAccuracySVM(train,train_labels,test,test_labels):
	clf=svm.SVC()
	clf.fit(train,train_labels)
	result=clf.predict(test)
	accuracy=0.0
	for i in result-test_labels:
		if i==0:
			accuracy+=1.0
	return 100.0*accuracy/len(test_labels)

def main():
	#trainFile="LSVT.csv"
	#trainLabels="LSVT.labels"
	#validFile="LSVT.csv"
	#validLabels="LSVT.labels"
	trainFile="arcene_train.data"
	trainLabels="arcene_train.labels"
	validFile="arcene_valid.data"
	validLabels="arcene_valid.labels"
	data=loadData(trainFile)
	classes=getClassIndex(trainLabels)
	KernelPCA = KPCA(data,classes,RBF)
	newData=KernelPCA.generateTrain(10)
	print checkAccuracySVM(newData,np.loadtxt(trainLabels,dtype=int),KernelPCA.generateTest(np.loadtxt(validFile)),np.loadtxt(validLabels,dtype=int))
	KernelLDA = KLDA(data,classes,RBF)
	newData=KernelLDA.generateTrain()
	print checkAccuracySVM(np.array([newData]).T,np.loadtxt(trainLabels,dtype=int),np.array([KernelLDA.generateTest(np.loadtxt(validFile))]).T,np.loadtxt(validLabels,dtype=int))

main()