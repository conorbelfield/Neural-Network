import numpy as np
from itertools import zip_longest

# filenames: [Train, Test]
BITMAPS = ["digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tra",
           "digit-recognition-examples/32x32-bitmaps/optdigits-32x32.tes"]

DOWNSAMPLES = ["digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tra",
               "digit-recognition-examples/8x8-integer-inputs/optdigits-8x8-int.tes"]


# reads in files in the format of the downsample files
# batchInput is a list of np arrays of type int
# batchOutput is a list of ints. entries correspond with each other
def readDownsamplesBatch(filename):
    batchInput = []
    batchOutput = []  
    file = open(filename)
    for line in file:
        batchPair = line.strip().split(',')
        # input is an array of the 32 block sums; output is the last integer
        batchInput.append(np.asarray([int(blockSum) for blockSum in batchPair[:-1]]))
        batchOutput.append(int(batchPair[-1]))    
    return batchInput, batchOutput

# auxiliary function: lets you read multiple lines of a file at a time
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

# reads in files in the format of the bitmap files
def readBitmapsBatch(filename):
    file = open(filename)
    i = 0
    # skip the first 3 lines of the file
    while i < 3:
        file.readline()  
        i+=1
    batchInput = []
    batchOutput = []      
    # read 33 lines at a time of the file
    for batchPair in grouper(file, 32 + 1):
        # create one giant string from the first 32 lines; strip whitespaces off each line
        batchPairInputString = "".join([bitString.strip() for bitString in batchPair[:-1]])
        # convert the string to a 1D array of type int
        batchPairInput = np.asarray(list(batchPairInputString), dtype=int)
        batchInput.append(batchPairInput)
        batchOutput.append(int(batchPair[-1].strip()))
    return batchInput, batchOutput

def batchImport(inputRepresentation):

    if inputRepresentation == "bitmaps":
        trainFilename, testFilename = BITMAPS
        xTrain, yTrain = readBitmapsBatch(trainFilename)
        xTest, yTest = readBitmapsBatch(testFilename)     
        
    elif inputRepresentation == "downsamples":
        trainFilename, testFilename = DOWNSAMPLES
        xTrain, yTrain = readDownsamplesBatch(trainFilename)
        xTest, yTest = readDownsamplesBatch(testFilename)      
    else:
        return -1
    print(yTest)
    return [xTrain, yTrain], [xTest, yTest]

#batchImport("downsamples")
batchImport("bitmaps")