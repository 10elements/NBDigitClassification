__author__ = "Dimitri Zhang"
import numpy as np
from matplotlib import pyplot as plt
import time
def simpleFeature(dataFileName, rowSize, colSize):
    """
    extract feature, a feature vector comprises of each pixel and number of circles
    :param dataFileName:file name of input dataM
    :param rowSize: row dimention of each single picture
    :param colSize:column dimention of each single picture
    :return: a matrix of which each row vector is the feature vector of each picture
    """
    datafile = open(dataFileName,"r")
    count = 0
    dataset = []
    new = True
    for line in datafile:
        count = count + 1
        if new:
            vec = []
            new = False
        temp = [0] * colSize
        for i,char in enumerate(line):
            if char == "#":
                temp[i] = 1
        vec = vec + temp
        if count == rowSize:
            digitMatrix = np.array(vec).reshape((rowSize, colSize))
            vec.append(circleNum(digitMatrix))
            dataset.append(vec)
            count = 0
            new = True
    return np.array(dataset)

def digitFeature(dataFileName, rowNum, colNum):
    datafile = open(dataFileName, "r")
    dataSet = []
    count = 0
    new = True
    for line in datafile:
        pre = None
        leftborders = 0
        start = 0
        length = 0
        firstBlack = True
        count += 1
        if new:
            featureVec = []
            digitMatrix = []
            new = False
        temp = [0] * colNum
        for i,char in enumerate(line):
            if char == "#" and firstBlack:
                firstBlack = False
                start = i
            if pre == " " and char == "#":
                temp[i] = 1
                leftborders += 1
                length = i - start + 1
            elif char == "#":
                temp[i] = 1
                length = i - start + 1
            pre = char
        featureVec.append(leftborders)
        featureVec.append(length)
        digitMatrix.append(temp)
        if count == rowNum:
            featureVec.append(circleNum(np.array(digitMatrix)))
            count = 0
            new = True
            dataSet.append(featureVec)
    return np.array(dataSet)

def neighbors(cell, M, N):
    """
    all neighbors of cell
    :param cell: input cell's indices, a tuple
    :param M: row dimention of digitMatrix
    :param N: column dimention of digitMatrix
    :return: a list of all neighbors of cell
    """
    allNeighbors = []
    row, column = cell
    if row > 0 and row < M - 1:
        allNeighbors.append((row + 1,column))
        allNeighbors.append((row - 1,column))
    elif row == M-1:
        allNeighbors.append((row - 1,column))
    elif row == 0:  
        allNeighbors.append((row + 1,column))

    if column > 0 and column < N - 1:
        allNeighbors.append((row,column + 1))
        allNeighbors.append((row,column - 1))
    elif column == N - 1:
        allNeighbors.append((row,column - 1))
    elif column == 0:
        allNeighbors.append((row,column + 1))
    return allNeighbors

def reachable(cell, Matrix, visited):
    """
    reachable neighbors of cell.
    :param cell: input cell' indices, a tuple.
    :param Matrix: the digit image.
    :param visited: visited list, a Matrix.
    :return:a list of all reachable neighbors' Indices
    """
    M, N = Matrix.shape
    n = neighbors(cell, M, N)
    result = []
    for item in n:
        if visited[item[0]][item[1]] == 0 and Matrix[item[0]][item[1]] == 0:# unvisited and unblocked
            result.append(item)
    return result

def argmax(a):
    """
    indices of the maximum value.
    :param a: input array
    :return:
    """
    m, n = a.shape
    i = np.argmax(a)
    row = int(i/n)
    column = i-row*n
    return row, column

def circleNum(imageMatrix):
    """
    number of circles of a digit image
    :param dataMatrix: digit image
    :return: number of circles
    """
    M, N = imageMatrix.shape
    visited = np.zeros((M, N), dtype = int)
    stack = [(0, 0)]
    visited[0][0] = 1
    circle = 0
    while True:
        while len(stack) != 0:# do DFS to find connected component
            current = stack[-1]
            available = reachable(current, imageMatrix, visited)
            if len(available) == 0:
                stack.pop()
            else:
                chosen = available[0]
                visited[chosen[0]][chosen[1]] = 1
                stack.append(chosen)
        temp = np.logical_xor(visited, imageMatrix)
        if np.logical_not(temp.all()):# if there are components unvisited
            circle += 1
            i, j = argmax(np.logical_not(temp))# do DFS in one of the unvisited components
            stack.append((i, j))
            visited[i][j] = 1
        else:# all components visited
            return circle

def faceFeature(dataFileName, rowNum, colNum):
    datafile = open(dataFileName,"r")
    dataSet = []
    count = 0
    blackNum = 0
    new = True
    for line in datafile:
        pre = None
        leftboeders = 0
        start = 0
        length = 0
        firstBlack = True
        count += 1
        if new:
            vec = []
            new = False
        for i,char in enumerate(line):
            if char == "#" and firstBlack:
                firstBlack = False
                start = i
            if pre == " " and char == "#":
                leftboeders += 1
                blackNum += 1
                length = i - start + 1
            elif char == "#":
                blackNum += 1
                length = i - start + 1
            pre = char
        vec.append(leftboeders)
        vec.append(length)
        if count == rowNum:
            vec.append(blackNum)
            dataSet.append(vec)
            count = 0
            blackNum = 0
            new = True
    return np.array(dataSet)

def loadLabels(labelFileName, size):
    label = []
    dsize = 0
    labelFile = open(labelFileName,"r")
    for line in labelFile:
        if dsize == size:
            break
        line = line.strip()
        label.append(line)
        dsize = dsize + 1
    return np.array(label)

def bernoulliTrain(dataSet, labelSet, k):
    dataSize = dataSet.shape[0]
    vecSize = dataSet.shape[1]
    labelNum = len(set(labelSet))
    numerator = np.array([[k] * vecSize] * labelNum)
    deno = np.array([2 * k] * labelNum)
    for i in range(dataSize):
        numerator[int(labelSet[i])] += dataSet[i]
        deno[int(labelSet[i])] += 1
    p = np.array( [ np.log( numerator[i] / deno[i] ) for i in range(labelNum) ] )
    pr = np.array( [ np.log( ( labelSet == str(i) ).sum() / float(dataSize) ) for i in range(labelNum) ] )
    return p,pr

def multinomialTrain(dataSet, labelSet, k):
    dataSize = dataSet.shape[0]
    vecSize  = dataSet.shape[1]
    # print(dataSize, vecSize)
    labelNum = len(set(labelSet.reshape((-1,)).tolist()))
    numerator = np.array([[k] * vecSize] * labelNum)
    deno = np.array([vecSize * k] * labelNum)
    for i in range(dataSize):
        numerator[labelSet[i]] += dataSet[i]
        deno[labelSet[i]] += sum(dataSet[i])
    p = np.array( [ np.log( numerator[i] / deno[i] ) for i in range(labelNum) ] )
    pr = np.array( [ np.log( ( labelSet == i ).sum() / float(dataSize) ) for i in range(labelNum) ] )
    return p,pr

def classify(testVec, proFeature, proClass):
    p = []
    labelSize = len(proClass)
    for i in range(labelSize):
        p.append(np.sum(testVec * proFeature[i]) + proClass[i])
    # sorted_p = sorted(p.items(), key = lambda x:x[1], reverse = True)
    result = 0
    maxpro = -float("inf")
    for j in range(labelSize):
        if p[j] > maxpro:
            result = j
            maxpro = p[j]
    return result

def digitRecognition():
    # digitTrainingData = simpleFeature("data/digitdata/trainingimages.txt", 28, 28)
    l=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    r=[]
    for traningSize in l:
        digitTrainingLabels = np.load('arrays/digits_training_y.npy').argmax(1).reshape((-1,1))
        size = digitTrainingLabels.shape[0]*traningSize
        digitTrainingLabels = digitTrainingLabels[0:size,]
        # print(digitTrainingLabels.shape)
        digitTrainingData = np.load('arrays/digits_training_x.npy')[0:size,]
        left_edge = np.load('arrays/digits_training_left_edge.npy')[0:size,]
        up_edge = np.load('arrays/digits_training_up_edge.npy')[0:size,]
        circle = np.load('arrays/digits_training_circles.npy')[0:size,]
        digitTrainingData = np.concatenate((digitTrainingData, left_edge, up_edge, circle), axis=1)
        digitTestData = np.load('arrays/digits_test_x.npy')
        t_left_edge = np.load('arrays/digits_test_left_edge.npy')
        t_up_edge = np.load('arrays/digits_test_up_edge.npy')
        t_circle = np.load('arrays/digits_test_circles.npy')
        digitTestData = np.concatenate((digitTestData, t_left_edge, t_up_edge, t_circle), axis=1)
        digitTestLabels = np.load('arrays/digits_test_y.npy').argmax(1).reshape((-1,1))
        dTestSize = digitTestData.shape[0]
        tic = time.time()
        dproF,dproC = multinomialTrain(digitTrainingData, digitTrainingLabels, 1)
        print(time.time()-tic)
        numMatch = 0
        for i in range(dTestSize):
            predicted = classify(digitTestData[i], dproF, dproC)
            # print("real value:%s, predicted value:%d"%(digitTestLabels[i], predicted))
            if digitTestLabels[i] == predicted:
                numMatch = numMatch + 1
        percentage = numMatch/dTestSize
        r.append(percentage)
        # print("numMatches:%d"%numMatch)
        # print("testSize:%d"%dTestSize)
        print("Accuracy rate: {:0.2f}%".format(percentage*100))
    print(len(l),len(r))
    plt.plot(l, r)
    plt.xlabel('percentage')
    plt.ylabel('Accuracy Rate')
    plt.show()

def faceRecognition():
    l=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    r=[]
    for traningSize in l:
        faceTrainData = np.load("arrays/faces_training_x.npy")
        size = faceTrainData.shape[0]*traningSize
        faceTrainData = faceTrainData[0:size,]
        circle = np.load("arrays/faces_training_circles.npy")[0:size,]
        faceTrainData = np.concatenate((faceTrainData, circle), axis=1)
        faceTrainLabels = np.load("arrays/faces_training_y.npy").reshape((-1, 1))
        faceTrainLabels = faceTrainLabels[0:size,]
        faceTestData = np.load("arrays/faces_test_x.npy")
        t_circle = np.load("arrays/faces_test_circles.npy")
        faceTestData = np.concatenate((faceTestData, t_circle), axis=1)
        faceTestLabels = np.load("arrays/faces_test_y.npy").reshape((-1,1))
        fTestSize = faceTestData.shape[0]
        tic = time.time()
        fproF, fproC = multinomialTrain(faceTrainData, faceTrainLabels, 1)
        print(time.time()-tic)
        numMatch = 0
        for i in range(fTestSize):
            predicted = classify(faceTestData[i], fproF, fproC)
            # print("real value:%s, predicted value:%d"%(faceTestLabels[i], predicted))
            if faceTestLabels[i] == predicted:
                numMatch = numMatch + 1
        percentage = numMatch/fTestSize
        r.append(percentage)
        print("Accuracy rate: {:0.2f}%".format(percentage*100))
    plt.plot(l,r)
    plt.xlabel("percentage")
    plt.ylabel("Accuracy Rate")
    plt.show()
def main():
    faceRecognition();
	# digitRecognition();
if __name__ == '__main__':
    main()
