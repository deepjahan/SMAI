import math
import random
import string
import numpy as np

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

# derivative of our sigmoid function
def der_sigmoid(y):
    return sigmoid(y)*(1.0-sigmoid(y))


# Generate an empty matrix
def generateMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0.0]*J)
    return m

class BNN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes resp.
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = generateMatrix(self.ni, self.nh)
        self.wo = generateMatrix(self.nh, self.no)

        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = 0.4*random.random() - 0.2
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = 4.0*random.random() - 2.0

        # last change in weights for momentum   
        self.ci = generateMatrix(self.ni, self.nh)
        self.co = generateMatrix(self.nh, self.no)

    def update(self, inputs):
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = der_sigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = der_sigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
    	print "Test Results:"
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def showWeights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print('\nOutput weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 10 == 0:
            	print "Done",100*i/iterations,"%"
            #    print('error %-.5f' % error)


def main():
    _pat=np.loadtxt("digit.txt",delimiter=",") #digit.txt contains train samples
    pat=[]
    for i in _pat:
        j=list(i)
        x=j.pop()
        if(int(x) == 0):
        	y=[1,0]
        elif(int(x)==7):
        	y=[0,1]
        pat.append([j,y])

    n = BNN(64, 4, 2)
    n.train(pat,40)
    n.showWeights()
    print "\n"
    #Some of the test samples
    n.test([ [[0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0], [1,0]],
    		 [[0,0,1,9,15,11,0,0,0,0,11,16,8,14,6,0,0,2,16,10,0,9,9,0,0,1,16,4,0,8,8,0,0,4,16,4,0,8,8,0,0,1,16,5,1,11,3,0,0,0,12,12,10,10,0,0,0,0,1,10,13,3,0,0],[1,0]],
    		 [[0,0,1,8,15,10,0,0,0,3,13,15,14,14,0,0,0,5,10,0,10,12,0,0,0,0,3,5,15,10,2,0,0,0,16,16,16,16,12,0,0,1,8,12,14,8,3,0,0,0,0,10,13,0,0,0,0,0,0,11,9,0,0,0],[0,1]]
    			 ])


main()
