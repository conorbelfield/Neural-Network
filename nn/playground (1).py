import math
import numpy as np
from ivies import Ivies
import time

WEIGHT_INIT_MAX_MAGNITUDE = 0.15
SIGMOID_SHIFT = 0.5
e = math.e


def trial(numEpochs, alpha, inputRepresentation, outputRepresentation):
    ivies = Ivies(inputRepresentation)
    numTotalTestPairs = len(ivies.xTest)
    startTime = time.time()
    W = trainWeights(numEpochs, ivies, outputRepresentation, alpha)
    numCorrect = testWeights(W, ivies)
    endTime = time.time()
    runTime = endTime - startTime
    return [[numEpochs, numTotalTestPairs, inputRepresentation, outputRepresentation], 
            [runTime, numCorrect, (numCorrect / numTotalTestPairs)* 100.0]]    

def testWeights(W, ivies):
    xTest = ivies.xTest
    yTest = ivies.yTest
    
    numCorrect = 0
    
    # output dim == 1
    if W.shape[0] == 1:
        for i in range(len(xTest)) :
            # i.e. T ~= O
            if yTest[i] == classifyN1(calcOutputN1(calcSigmoidInput(xTest[i], W))):
                numCorrect += 1         

    # output dim == 10
    elif W.shape[0] == 10:
        for i in range(len(xTest)):
            if yTest[i] == classifyN10(calcOutputN10(calcSigmoidInput(xTest[i], W))):
                numCorrect += 1
    
    return numCorrect   

def trainWeights(numEpochs, ivies, outputDim, alpha):
    inputDim = 8**2
    if ivies.inputRepresentation == "bitmaps":
        inputDim = 32**2
        
    W = constructWeights(inputDim, outputDim)
    #print(W)
    if outputDim == 1:
        for epoch in range(numEpochs):
            W = singleEpochBatchN1(W, ivies.xTrain, ivies.yTrain, inputDim+1, outputDim, alpha)
            if epoch % 5 == 0:
                correct = testWeights(W, ivies)
                print("Epoch: ", epoch, "%= : ", 100.0 * correct / len(ivies.xTest))
     #       print(W)
    elif outputDim == 10:
        for epoch in range(numEpochs):
            W = singleEpochBatchN10(W, ivies.xTrain, ivies.yTrain, inputDim+1, outputDim, alpha)
            if epoch % 5 == 0:
                correct = testWeights(W, ivies)
                print("Epoch: ", epoch, "%= : ", 100.0 * correct / len(ivies.xTest))            
      #      print(W)
    return W

def singleEpochBatchN1(W, batchInput, batchOutput, inputDim, outputDim, alpha):
    batchSz = len(batchInput)
    for i in range(batchSz):
        iVec = batchInput[i]
        T = batchOutput[i]
        W = singleEpochBatchPairN1(iVec, T, W, inputDim, outputDim, alpha)
        #print(W)
    return W

def singleEpochBatchN10(W, batchInput, batchOutput, inputDim, outputDim, alpha):
    batchSz = len(batchInput)
    for i in range(batchSz):
        iVec = batchInput[i]
        T = batchOutput[i]
        W = singleEpochBatchPairN10(iVec, T, W, inputDim, outputDim, alpha)
    return W
    
def singleEpochBatchPairN1(iVec, T, W, inputDim, outputDim, alpha):
    sigmoidInput = calcSigmoidInput(iVec, W)
    O = calcOutputN1(sigmoidInput)
    Err = calcErrorN1(T, O)
    W = updateN1(iVec, W, Err, alpha, sigmoidInput, inputDim, outputDim)    
    return W

def singleEpochBatchPairN10(iVec, T, W, inputDim, outputDim, alpha):
    sigmoidInput = calcSigmoidInput(iVec, W)
    O = calcOutputN10(sigmoidInput)
    Err = calcErrorN10(T, O)
    W = updateN10(iVec, W, Err, alpha, sigmoidInput, inputDim, outputDim)    
    return W

def classifyN1(O):
    # mult output by 10; round; cast to int
    classification = int(np.round(O * 10))
    if classification == 10:
        classification = 9
    return classification

#should this be + 1 because is an index returned from argmax?
def classifyN10(O):
    # O = calcOutputN10(inputVec, weights)
    return np.argmax(O)

def constructWeights(numInputs, numOutputs):
    # matrix of random doubles from 0 to 1; plus a weight for the bias node
    W = np.random.rand(numOutputs, numInputs + 1)
    W = (W - 0.5) * 2 * WEIGHT_INIT_MAX_MAGNITUDE
    return W

def sigmoid(x):
    return 1 / (1 + e**(SIGMOID_SHIFT - x))

def derivSigmoid(x):
    return e**(SIGMOID_SHIFT + x) / ((e**(SIGMOID_SHIFT)) + (e**x))**2

#dotProd weights, input; 
def calcSigmoidInput(inputVec, weights):
    return np.dot(weights, inputVec)

def calcOutputN1(sigmoidInput):
    #sigmoid of dotProd;
    return sigmoid(sigmoidInput)[0]

def calcOutputN10(sigmoidInput):
    return sigmoid(sigmoidInput)

def calcErrorN1(T, O):
    return (T / 10) - O

# array of length 10
def calcErrorN10(T, O):
    vecT = np.zeros(10)
    vecT[T] = 1
    return vecT - O
    
# i = inputIndex; W = weights; iVec = input
def updateN1(iVec, W, Err, alpha, sigmoidInput, inputDim, outputDim):
    for i in range(inputDim):
        W[0][i] = W[0][i] + alpha * iVec[i] * Err * derivSigmoid(sigmoidInput)
        #print(i)
        #print(W)
    return W

def updateN10(iVec, W, Err, alpha, sigmoidInput, inputDim, outputDim):
    for i in range(outputDim):
        for j in range(inputDim):
            W[i][j] = W[i][j] + alpha * iVec[j] * Err[i] * derivSigmoid(sigmoidInput[i])
    return W



def runTrials(numEpochs):
    inputs = ["downsamples", "bitmaps"]
    outputs = [1, 10]
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    data = []
    for i in range(len(inputs)):
        for j in range(len(outputs)):
            for k in range(len(alphas)):            
                trialData = trial(numEpochs, alphas[k], inputs[i], outputs[j])
                data.append(trialData)                    
                print("alpha=", alphas[k], ", input=", inputs[i], ", output=", outputs[j])
                print("time=", trialData[1][0], ", %=", trialData[1][2])      
    return trialData

runTrials(50)