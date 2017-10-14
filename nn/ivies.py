import numpy as np
from itertools import zip_longest

# filenames: [Train, Test]
BITMAPS = ["digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra",
           "digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes"]

DOWNSAMPLES = ["digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra",
               "digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes"]

class Ivies(object):
    
    def __init__(self, inputRepresentation):
        xTrain, yTrain, xTest, yTest = self.batchImport(inputRepresentation)
        
        self.inputRepresentation = inputRepresentation
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.xTest = xTest
        self.yTest = yTest
        

    # reads in files in the format of the downsample files
    # batchInput is a list of np arrays of type int
    # batchOutput is a list of ints. entries correspond with each other
    def readDownsamplesBatch(self, filename):
        batchInput = []
        batchOutput = []  
        file = open(filename)
        for line in file:
            batchPair = line.strip().split(',')
            # input is an array of the 32 block sums; output is the last integer
            batchPairInput = np.asarray([int(blockSum) for blockSum in batchPair[:-1]])
            # add bias node
            batchPairInput = np.concatenate((batchPairInput, [1]), axis=0)
            batchInput.append(batchPairInput)
            batchOutput.append(int(batchPair[-1]))    
        return batchInput, batchOutput
    
    # auxiliary function: lets you read multiple lines of a file at a time
    def grouper(self, iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)
    
    # reads in files in the format of the bitmap files
    def readBitmapsBatch(self, filename):
        file = open(filename)
        i = 0
        # skip the first 3 lines of the file
        while i < 3:
            file.readline()  
            i+=1
        batchInput = []
        batchOutput = []      
        # read 33 lines at a time of the file
        for batchPair in self.grouper(file, 32 + 1):
            # create one giant string from the first 32 lines; strip whitespaces off each line
            batchPairInputString = "".join([bitString.strip() for bitString in batchPair[:-1]])
            # convert the string to a 1D array of type int
            batchPairInput = np.asarray(list(batchPairInputString), dtype=int)
            # add bias node
            batchPairInput = np.concatenate((batchPairInput, [1]), axis=0)
            batchInput.append(batchPairInput)
            batchOutput.append(int(batchPair[-1].strip()))
        return batchInput, batchOutput
    
    def batchImport(self, inputRepresentation):
    
        if inputRepresentation == "bitmaps":
            trainFilename, testFilename = BITMAPS
            xTrain, yTrain = self.readBitmapsBatch(trainFilename)
            xTest, yTest = self.readBitmapsBatch(testFilename)     
            
        elif inputRepresentation == "downsamples":
            trainFilename, testFilename = DOWNSAMPLES
            xTrain, yTrain = self.readDownsamplesBatch(trainFilename)
            xTest, yTest = self.readDownsamplesBatch(testFilename)      
        else:
            return -1
        return xTrain, yTrain, xTest, yTest