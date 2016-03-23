import numpy as np

def loadData(file,file2):
	data=np.loadtxt(file)
	d=len(data[0])
	labels=np.loadtxt(file2)
	classes={}
	for i in range(len(labels)):
		if labels[i] not in classes.keys():
			classes[labels[i]]=[np.array(data[i])]
		else:
			classes[labels[i]].append(np.array(data[i]))
	m={}
	for i in classes.keys():
		classes[i]=np.array(classes[i])
		m[i]=classes[i].mean(0)
	Sw=np.zeros((d,d))
	for i in classes.keys():
		for j in classes[i]:
			Sw+=np.dot(np.array([j-m[i]]).T,np.array([j-m[i]]))
	Sb=np.dot(np.array([m[classes.keys()[0]]-m[classes.keys()[1]]]).T,np.array([m[classes.keys()[0]]-m[classes.keys()[1]]]))
	print Sw
	lda_eig_val,lda_eig_vec = np.linalg.eigh(np.dot(np.linalg.inv(Sw-0.1*np.identity(d)),Sb))
	lda_eig=zip(lda_eig_val,lda_eig_vec.T)
	lda_eig.sort(reverse=True)

	lst=[]
	for i in range(d):
		lst.append(np.mean(data[:,i]))
	mean=np.array(lst)
	"""scatter=np.zeros((d,d))
	for i in range(len(data)):
		a=data[i]-mean
		print i
		scatter+= np.array([a]).T.dot(np.array([a]))"""
	lst=[]
	for i in range(d):
		lst.append(data[:,i])
	cov=np.cov(lst)
	eig_val_cov, eig_vec_cov = np.linalg.eigh(cov)
	eig=zip(np.array(eig_val_cov),np.array(eig_vec_cov).T)
	eig.sort(reverse=True)
	return (data,eig,lda_eig[0][1])

def newSubspacePCA(data,eig,k):
	W=[]
	for i in range(k):
		W.append(eig[i][1])
	W=np.array(W)
	return np.dot(data,W.T)

def newSubspaceLDA(data,w):
	W=[]
	for i in data:
		W.append(np.dot(w,i.T))
	W=np.array(W)
	return W

class gaussBayes(object):
	def __init__(self, train,classes):
		self.train=train
		self.classes=classes
		self.d=len(train)
		self.gs={}
		self.prior={}
	def trainData(self):
		for w in set(classes):
			lst=[]
			tot=0.0
			for i in range(self.d):
				avg=0.0
				tot=0.0
				for j in range(len(self.train[0])):
					if self.classes[j]==w:
						avg+=self.train[i][j]
						tot+=1
				avg/=tot
				lst.append([avg])
			mean=np.array(lst)
			lst=[]
			for i in range(self.d):
				lst2=[]
				for j in range(len(self.train[0])):
					if self.classes[j]==w:
						lst2.append(self.train[i][j])
				lst.append(np.array(lst2))
			cov=np.cov(lst)
			self.gs[w]=(mean,cov)
			self.prior[w]=tot/len(self.train[0])
	def testData(self,test):
		max_lik=0.0
		final_class=classes[0]
		for w in set(classes):
			likelihood=pow(math.e,-0.5*((test-self.gs[0]).T.dot(np.linalg.inv(self.gs[1]))).dot(test-self.gs[0]))/pow(np.linalg.det(self.gs[1]),0.5)*self.prior[w]
			if likelihood>max_lik:
				max_lik=likelihood
				final_class=w
		return final_class
	def accuracy(self,test,labels):
		accuracy=0.0
		LL=len(labels)
		for i in range(LL):
			if(self.testData(test[i])==labels[i]):
				accuracy+=1.0
		print 100.0*accuracy/LL

def main():
	data=loadData("./Arcene/arcene_train.data","./Arcene/arcene_train.labels")
	classes=np.loadtxt("./Arcene/arcene_train.labels")
	newDataPCA=newSubspacePCA(data[0],data[1],1000)
	newDataLDA=newSubspaceLDA(data[0],data[2])

	validData=np.loadtxt("./Arcene/arcene_valid.data")
	validClasses=np.loadtxt("./Arcene/arcene_valid.labels")
	newValidDataPCA=newSubspacePCA(validData,data[1],1000)
	newValidDataLDA=newSubspaceLDA(validData,data[2])

	B_PCA=gaussBayes(newDataPCA,classes)
	B_PCA.accuracy(newValidDataPCA,validClasses)
	B_LDA=gaussBayes(newDataLDA,classes)
	B_LDA.accuracy(newValidDataLDA,validClasses)

main()