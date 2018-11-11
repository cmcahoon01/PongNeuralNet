import numpy as np
import pylab as plt
import copy

class NeuralNetwork():

    def __init__(self,numInputs,numHiddenNodes,numHiddenLayers,numOutputs):
        self.inputs=np.zeros(numInputs)
        self.HiddenNodes=np.zeros([numHiddenLayers,numHiddenNodes])
        self.outputs=np.zeros(numOutputs)
        self.inWeights=np.zeros([numHiddenNodes,numInputs])
        self.hiddenWeights=np.zeros([numHiddenLayers-1,numHiddenNodes,numHiddenNodes])
        self.outWeights=np.zeros([numOutputs,numHiddenNodes])
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        self.score=0

    def randomizeWeights(self):
        self.inWeights=np.random.rand(len(self.HiddenNodes[0]),len(self.inputs))*2-1
        self.hiddenWeights=np.random.rand(len(self.HiddenNodes)-1,len(self.HiddenNodes[0]),len(self.HiddenNodes[0]))*2-1
        self.outWeights=np.random.rand(len(self.outputs),len(self.HiddenNodes[-1]))*2-1

    def propagate(self,givenInputs):
        self.inputs=np.array(givenInputs)
        self.HiddenNodes[0]=self.inWeights.dot(self.inputs)
        for i in range(1,len(self.HiddenNodes)):
            self.HiddenNodes[i]=self.hiddenWeights[i-1].dot(self.HiddenNodes[i-1])
        self.outputs=self.outWeights.dot(self.HiddenNodes[-1])
        return self.outputs.tolist()

    def mutate(self,mutationChance,mutationRate):
        for i in range(len(self.inWeights)):
            for j in range(len(self.inWeights[i])):
                if np.random.random()<mutationChance:
                    self.inWeights[i][j]=self.adjust(self.inWeights[i][j],mutationRate)
        for i in range(len(self.hiddenWeights)):
            for j in range(len(self.hiddenWeights[i])):
                if np.random.random()<mutationChance:
                    self.hiddenWeights[i][j]=self.adjust(self.hiddenWeights[i][j],mutationRate)
        for i in range(len(self.outWeights)):
            for j in range(len(self.outWeights[i])):
                if np.random.random()<mutationChance:
                    self.outWeights[i][j]=self.adjust(self.outWeights[i][j],mutationRate)

    def adjust(self,toAdj,mutationRate):
        randChange=np.random.random()**(1/mutationRate)
        if np.random.random()>0.5:
            randChange*=-1
        return toAdj+randChange

    def clone(self):
        new=copy.deepcopy(self)
        new.clrScore()
        return new

    def setScore(self,n):
        self.score=n

    def addScore(self,n):
        self.score+=n

    def clrScore(self):
        self.score=0

    def getScore(self):
        return self.score

class sigmaNet(NeuralNetwork):
    def __init__(self,numInputs,numHiddenNodes,numHiddenLayers,numOutputs):
        self.inputs=np.zeros(numInputs)
        self.HiddenNodes=np.zeros([numHiddenLayers,numHiddenNodes])
        self.outputs=np.zeros(numOutputs)
        self.inWeights=np.zeros([numHiddenNodes,numInputs+1])
        self.hiddenWeights=np.zeros([numHiddenLayers-1,numHiddenNodes,numHiddenNodes+1])
        self.outWeights=np.zeros([numOutputs,numHiddenNodes+1])
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        self.score=0

    def randomizeWeights(self):
        self.inWeights=np.random.rand(len(self.HiddenNodes[0]),len(self.inputs)+1)*2-1
        self.hiddenWeights=np.random.rand(len(self.HiddenNodes)-1,len(self.HiddenNodes[0]),len(self.HiddenNodes[0])+1)*2-1
        self.outWeights=np.random.rand(len(self.outputs),len(self.HiddenNodes[-1])+1)*2-1

    def sigmoid(self,arr):
        return (arr+abs(arr))/2
        return 1 / (1 + np.exp(-arr))

    def propagate(self,givenInputs,p=False):
        self.inputs=np.array(givenInputs)
        self.inputs=np.append(self.inputs,1)
        if p:
            print(self.inputs)
        self.HiddenNodes[0]=self.inWeights.dot(self.inputs)
        if p:
            print(self.HiddenNodes[0])
        self.HiddenNodes[0]=self.sigmoid(self.HiddenNodes[0])
        if p:
            print(self.HiddenNodes[0])
        for i in range(1,len(self.HiddenNodes)):
            self.HiddenNodes[i]=self.hiddenWeights[i-1].dot(np.append(self.HiddenNodes[i-1],[1]))
            self.HiddenNodes[i]=self.sigmoid(self.HiddenNodes[i])
        self.outputs=self.outWeights.dot(np.append(self.HiddenNodes[-1],[1]))
        if p:
            print(self.outputs)
        self.outputs=self.sigmoid(self.outputs)
        return self.outputs.tolist()

    def adjust(self,toAdj,mutationRate):
        toAdj=toAdj+(np.random.random()*2-1)**(1/mutationRate)
        toAdj*=np.random.choice([-1,1])
        return toAdj
