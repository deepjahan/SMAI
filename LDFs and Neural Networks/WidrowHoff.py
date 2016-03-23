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

def widrow_hoff_main():
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
	b=[1.0]*n
	i=0
	eeta=1
	count=0
	theta=10000000000000000000000000
	dot=np.dot(inplist[i],a)
	upd=pow(abs(b[i]-dot),1)*inplist[i]*eeta
	while(np.dot(upd,upd)<theta*theta):
		misclassified=0
		if( inpdict[(inplist[i][0],inplist[i][1])] == 1 ):
			if dot>-b[i]:
				misclassified=-1
		else:
			if dot<b[i]:
				misclassified=1
		if misclassified != 0:
			a+=(misclassified*upd)
			count=0
		else:
			count+=1
		if count==n:
			break
		i+=1
		i%=n
		dot=np.dot(inplist[i],a)
		upd=pow(abs(b[i]-dot),1/2)*inplist[i]*eeta
	return a