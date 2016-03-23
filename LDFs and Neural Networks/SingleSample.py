import numpy as np
import random

def loadInput(filename):
	f=open(filename,"r")
	inp=np.loadtxt(filename,dtype={'names': ('x','y','class'),'formats':(np.float, np.float, np.int)})
	inplist = list(inp)
	random.shuffle(inplist)
	inpdict={}
	for i in inplist:
		inpdict[(i[0],i[1])]=i[2]
	return inpdict

def single_sample_main():
	inpdict=loadInput("input.txt")
	inpkeys=inpdict.keys()
	inplist=[]
	for i in inpkeys:
		j=list(i)
		j.append(1.0)
		inplist.append(np.array(j))
	n=len(inpkeys)
	dim=len(inpkeys[0])
	a=np.random.rand(1,dim+1)[0] * 20 - 10
	i=0
	eeta=1
	count=0
	while(1):
		dot=np.dot(inplist[i],a)
		misclassified=0
		if dot>0.0:
			if( inpdict[(inplist[i][0],inplist[i][1])] == 1 ):
				misclassified=-1
		elif dot<0.0:
			if( inpdict[(inplist[i][0],inplist[i][1])] == 2 ):
				misclassified=1
		if misclassified != 0:
			a+=(misclassified*eeta*inplist[i])
			count=0
		else:
			count+=1
		if count==n:
			break
		#print a
		i+=1
		i%=n
	return a
