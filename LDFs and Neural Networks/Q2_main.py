import numpy as np
import matplotlib.pyplot as plt
import random
from SingleSample import single_sample_main
from SingleSampleMargin import single_sample_margin_main
from RelaxationMargin import relaxation_margin_main
from WidrowHoff import widrow_hoff_main

def loadInput(filename):
	f=open(filename,"r")
	inp=np.loadtxt(filename,dtype={'names': ('x','y','class'),'formats':(np.float, np.float, np.int)})
	inplist = list(inp)
	random.shuffle(inplist)
	inpdict={}
	for i in inplist:
		inpdict[(i[0],i[1])]=i[2]
	return inpdict

def main():
	inpdict=loadInput("input.txt")
	inpkeys=inpdict.keys()
	x=[]
	y=[]
	for i in inpkeys:
		if inpdict[i]==1:
			x.append(i[0])
			y.append(i[1])
	plt.plot(x,y,'ro',color="red")
	x=[]
	y=[]
	for i in inpkeys:
		if inpdict[i]==2:
			x.append(i[0])
			y.append(i[1])
	plt.plot(x,y,'ro',color="green")
	x=[0.0,10.0]	
	a=single_sample_main()
	if(abs(a[1])>0.000001):
		y1=(-a[0]*x[0]-a[2])/a[1]
		y2=(-a[0]*x[1]-a[2])/a[1]
	else:
		y=[0.0,0.0]
	y=[y1,y2]
	plt.plot(x,y,label="Single Sample")
	a=single_sample_margin_main()
	if(abs(a[1])>0.000001):
		y1=(-a[0]*x[0]-a[2])/a[1]
		y2=(-a[0]*x[1]-a[2])/a[1]
	else:
		y=[0.0,0.0]
	y=[y1,y2]
	plt.plot(x,y,label="Single Sample with Margin")
	a=relaxation_margin_main()
	if(abs(a[1])>0.000001):
		y1=(-a[0]*x[0]-a[2])/a[1]
		y2=(-a[0]*x[1]-a[2])/a[1]
	else:
		y=[0.0,0.0]
	y=[y1,y2]
	plt.plot(x,y,label="Relaxation with Margin")
	a=widrow_hoff_main()
	if(abs(a[1])>0.000001):
		y1=(-a[0]*x[0]-a[2])/a[1]
		y2=(-a[0]*x[1]-a[2])/a[1]
	else:
		y=[0.0,0.0]
	y=[y1,y2]
	plt.plot(x,y,label="Widrow-Hoff")
	plt.axis([0,10,0,10])
	plt.legend()
	plt.show()

main()