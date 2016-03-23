import numpy as np
import random

"""This is incomplete"""
def min(data,D1,D2,Distance):
	d_min=Distance[D1[0]][D2[0]]
	for i in D1:
		for j in D2:
			if Distance[i][j]<d_min:
				d_min=Distance[i][j]
	return d_min

def max(data,D1,D2,Distance):
	d_max=Distance[D1[0]][D2[0]]
	for i in D1:
		for j in D2:
			if Distance[i][j]>d_max:
				d_max=Distance[i][j]
	return d_max

def avg(data,D1,D2,Distance):
	d_avg=0.0
	for i in D1:
		for j in D2:
			d_avg+=Distance[i][j]
	d_avg/=(len(D1)*len(D2))
	return d_avg

def mean(data,D1,D2,Distance):
	d_mean=np.mean(data[D1],0)-np.mean(data[D2],0)
	return sqrt(np.dot(d_mean,d_mean))




class Cluster(object):
	def __init__(self, data):
		self.data=data
		self.n=len(data)
		self.Distance=np.zeros((self.n,self.n))
		for i in range(self.n):
			for j in range(i+1,self.n):
				dist=data[i]-data[j]
				self.Distance[i][j]=sqrt(np.dot(dist,dist))
				self.Distance[j][i]=self.Distance[i][j]
	def Agglomerative(self,c,criterion):
		self.aggC=[]
		Matrix=np.zeros((self.n,self.n))
		for i in range(self.n):
			self.aggC.append([i])
		for i in range(self.n):
			for j in range(i+1,self.n):
				Matrix[i][j]=criterion(self.data,self.aggC[i],self.aggC[j],self.Distance)
				Matrix[j][i]=Matrix[i][j]
		i=np.max(Matrix)
		for j in range(self.n):
			Matrix[j][j]=i+1
		d=n;
		while(c<d):
			d-=1
			i=np.argmin(Matrix)
			x=i/self.n
			y=i%self.n
			self.argC[x]=self.argC[x]+self.argC[y]
			self.argC.remove(self.argC[y])
			Matrix_dum=np.zeros((d,d))
			for i in range(d):
				for j in range(i+1,d):
					if i==x || j==x:
						Matrix_dum[i][j]=criterion(self.data,self.aggC[i],self.aggC[j],self.Distance)
						Matrix_dum[j][i]=Matrix_dum[i][j]
					elif i<y && j<y:
						Matrix_dum[i][j]=Matrix[i][j]
						Matrix_dum[j][i]=Matrix[i][j]
					elif i>=y && j<y:
						Matrix_dum[i][j]=Matrix[i+1][j]
						Matrix_dum[j][i]=Matrix_dum[i][j]
					elif i>=y && j>=y:
						Matrix_dum[i][j]=Matrix[i+1][j+1]
						Matrix_dum[j][i]=Matrix_dum[i][j]
					elif i<y && j>=y:
						Matrix_dum[i][j]=Matrix[i][j+1]
						Matrix_dum[j][i]=Matrix_dum[i][j]
			Matrix=Matrix_dum
		return self.argC

	def fuzzyMean(P,c,data,n):
		means=[]
		for j in range(c):
			tot=0.0
			for i in range(n):
				tot+=P[j][i]
			mean=0.0
			for i in range(n):
				mean+=(P[j][i]/tot)*data[i]
			means.append(mean)
		return np.array(means)
	def fuzzyK(self,c,cent=fuzzyMean):
		centroids=np.array(random.sample(self.data,c))
		P=np.zeros((c,self.n))
		while(1):
			for i in range(c):
				evidence=0.0
				for j in range(self.n):
					vec=self.data[j]-centroids[i]
					evidence+=(1.0/sqrt(np.dot(vec,vec)))
				for j in range(self.n):
					vec=self.data[j]-centroids[i]
					P[i][j]=1.0/(evidence*sqrt(np.dot(vec,vec)))
			newCentroids=cent(P,c,self.data,self.n)
			if np.all(np.isclose(newCentroids-centroids,0)):
				break
			centroids=newCentroids
		return centroids

	def medoid(data,c,Distance):
		means=[]
		for i in range(c):
			mean=data[i][0]
			dist=0.0
			for k in data[i]:
				dist+=Distance[0][k]
			d_min=dist
			for j in data[i][1:]:
				dist=0.0
				for k in data[i]:
					dist+=Distance[j][k]
				if dist<d_min:
					d_min=dist
					mean=j
			means.append(mean)
		return np.array(means)
	def medoidK(self,c,cent=medoid):
		centroids=np.array(random.sample(range(self.n),c))
		while(1):
			ans=[]
			for i in range(c):
				ans.append([])
			for i in range(self.n):
				clss=0
				d_min=Distance[i][centroids[0]]
				for j in range(1,c):
					if Distance[i][centroids[j]]<d_min:
						d_min=Distance[i][centroids[j]]
						clss=j
				ans[clss].append(self.data[i])
			newCentroids=cent(np.array(ans),c,self.Distance)
			if np.all(np.isclose(newCentroids-centroids,0)):
				break
			centroids=newCentroids
		return self.data[centroids]

