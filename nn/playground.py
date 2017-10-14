import math
import numpy as np
from ivies import Ivies

WEIGHT_INIT_MAX_MAGNITUDE = 0.15
SIGMOID_SHIFT = 0.5
e = math.e
alpha = 0.5

def itsLit(inputRepresentation, inputDim, outputDim):
    
    inputRepresentation = Ivies(inputRepresentation)
    W = constructWeights(inputDim, outputDim)
    if outputDim == 1:  
        for i in range(100):
            W = singleEpochBatchN1(W, inputRepresentation.xTrain, inputRepresentation.yTrain)
    else:
        for i in range(100):
            W = singleEpochBatchN10(W, inputRepresentation.xTrain, inputRepresentation.yTrain)
        

    # classification after one epoch    
    inputVec = inputRepresentation.xTrain[0]
    T = inputRepresentation.yTrain[0]    
    sigmoidInput = calcSigmoidInput(inputVec, W)    
    if outputDim == 10:
        O = calcOutputN10(sigmoidInput)
        classification = classifyN10(O)
    else:
        O = calcOutputN1(sigmoidInput)
        classification = classifyN1(O)
    print(classification)
    print(T)
    return W
    
def singleEpochBatchN1(W, batchInput, batchOutput):
    batchSz = len(batchInput)
    for i in range(batchSz):
        iVec = batchInput[i]
        T = batchOutput[i]
        W = singleEpochBatchPairN1(iVec, T, W)
    return W

def singleEpochBatchN10(W, batchInput, batchOutput):
    batchSz = len(batchInput)
    for i in range(batchSz):
        iVec = batchInput[i]
        T = batchOutput[i]
        W = singleEpochBatchPairN10(iVec, T, W)
    return W
    
def singleEpochBatchPairN1(iVec, T, W):
    sigmoidInput = calcSigmoidInput(iVec, W)
    O = calcOutputN1(sigmoidInput)
    Err = calcErrorN1(T, O)
    W = updateN1(iVec, W, Err, alpha, sigmoidInput)    
    return W

def singleEpochBatchPairN10(iVec, T, W):
    sigmoidInput = calcSigmoidInput(iVec, W)
    O = calcOutputN10(sigmoidInput)
    Err = calcErrorN10(T, O)
    W = updateN10(iVec, W, Err, alpha, sigmoidInput)    
    return W

def classifyN1(O):
    # mult output by 10; round; cast to int
    classification = int(np.round(O * 10))
    if classification == 10:
        classification = 9
    return classification

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
    return e**(SIGMOID_SHIFT - x) / (1 + e**(SIGMOID_SHIFT - x))**2

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
    T = np.zeros(10)
    T[classifyN10(O)] = 1
    return T - O
    
# i = inputIndex; W = weights; iVec = input
def updateN1(iVec, W, Err, alpha, sigmoidInput):
    for i in range(len(W)):
        W[i] = W[i] + alpha * iVec[i] * Err * derivSigmoid(sigmoidInput)
    return W

def updateN10(iVec, W, Err, alpha, sigmoidInput):
    for i in range(10):
        for j in range(len(W)):
            W[i][j] = W[i][j] + alpha * iVec[j] * Err[i] * derivSigmoid(sigmoidInput[i])
    return W

itsLit("downsamples", 8**2, 10)
#itsLit("bitmaps", 32**2, 10)

