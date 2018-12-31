import copy
import math
import operator
import time
of = open("./BEOutput.txt", "w+")
def extractData(file):
    data = []
    with open(file) as f:
        for i in f.readlines():
            try:
                dataLine = i.strip(" ")
                dataLine = [float(j) for j in dataLine.split()]
                dataLine[0] = int(dataLine[0])
                data.append(dataLine)
            except ValueError :
                print("Error Reading file")
    return data

def normalizeData(data, numFeatures, records):
    dataNorm = data
    avg = [0.00000]*(numFeatures)
    sdev = copy.deepcopy(avg)
    for dataLine in dataNorm:
        for index, y in enumerate(dataLine[1:]):
            avg[index] += y

    for index, x in enumerate(avg):
        avg[index] = avg[index]/records

    for i,dataLine in enumerate(dataNorm):
        for index, y in enumerate(dataLine[1:]):
            sdev[index] += pow((y-avg[index]),2)
    for i, value in enumerate(sdev):
        sdev[i] = math.sqrt(value/records)

    for xindex, dataLine in enumerate(dataNorm):
        for yindex, value in enumerate(dataLine[1:]):
            dataNorm[xindex][yindex+1] = (value - avg[yindex])/sdev[yindex]

    return dataNorm

def findDistance(trainInst, testInst, dataFlags):
    dist = 0
    for index, flag in enumerate(dataFlags):
        if flag:
            dist += pow((testInst[index]-trainInst[index]),2)
    return dist

def findNN(trainSet, testSet, dataFlags):
    distMeasures = []
    neighbor = []
    for x in trainSet:
        dist = findDistance(x, testSet, dataFlags)
        distMeasures.append((x,dist))
    distMeasures.sort(key=operator.itemgetter(1))
    return distMeasures[0][0]

def calcAccuracy(data, dataFlags):
    accuracy = 0.00000
    for i in range(len(data)):
        trainSet = list(data)
        testSet = trainSet.pop(i)
        nearestNeighbor = findNN(trainSet, testSet, dataFlags)
        if (nearestNeighbor[0] == testSet[0]):
            accuracy += 1
    accuracy = accuracy*100/len(data)
    return accuracy

def calcAccuracyGSpecial(data, dataFlags,leastMisses):
    accuracy = 0.00000
    miss = 0
    for i in range(len(data)):
        trainSet = list(data)
        testSet = trainSet.pop(i)
        nearestNeighbor = findNN(trainSet, testSet, dataFlags)
        if (nearestNeighbor[0] == testSet[0]):
            accuracy += 1
        else:
            miss += 1
            if miss > leastMisses and leastMisses != -1:
                return (0, leastMisses)
    accuracy = accuracy*100/len(data)
    return [accuracy,miss]

def allFeatureKNN(data,numFeatures,records):
    dataFlags = [1]*(len(data[0])-1)
    dataFlags.insert(0,0)
    accuracy = calcAccuracy(data, dataFlags)
    print("Running nearest neighbor with all",numFeatures,"features, using \"leaving-one-out\" evaluation, I get an accuracy of",accuracy,"%\n")

def createDataFlag(featureFlags, currentFeatures, x, selectType):
    dataFlags = [0]*(len(featureFlags)+1)
    for i in currentFeatures:
        dataFlags[i] = 1
    if(selectType == 1):
        dataFlags[x] = 1
    else:
        dataFlags[x] = 0
    return dataFlags

def printFeatureSet(tempFeatures,accuracy,pType):

    if pType == 1:
        if (len(tempFeatures) == 1):
            print("\tUsing feature(s) {",tempFeatures[0],"} accuracy is",accuracy,"%")
            # x = "\tUsing feature(s) {"+str(tempFeatures[0])+"} accuracy is "+str(accuracy)+"%\n"
            # of.write(''.join(x))
        else:
            print("\tUsing features(s) {",','.join(str(i) for i in tempFeatures),"} accuracy is",accuracy,"%")
            # x = "\tUsing features(s) {",','.join(str(i) for i in tempFeatures),"} accuracy is "+str(accuracy)+"%\n"
            # of.write(''.join(x))
    else:
        if (len(tempFeatures) == 1):
            print("Feature set {",tempFeatures[0],"} was best accuracy is",accuracy,"%\n")
            # x = "Feature set {"+str(tempFeatures[0])+"} was best accuracy is "+str(accuracy)+"%\n"
            # of.write(''.join(x))
        else:
            print("Features set {",','.join(str(i) for i in tempFeatures),"} was best accuracy is",accuracy,"%\n")
            # x = "Features set {",','.join(str(i) for i in tempFeatures),"} was best accuracy is "+str(accuracy)+"%\n"
            # of.write(''.join(x))

def findBestFeatureFS(data, featureFlags, currentFeatures, bestAcc):
    featNUsed = [i for i in featureFlags if i not in currentFeatures]
    featureAcc = [0.00000] * len(featNUsed)
    i = 0
    accuracy = 0
    for x in featNUsed:
        dataFlags = createDataFlag(featureFlags, currentFeatures, x,1)
        featureAcc[i] = calcAccuracy(data,dataFlags)
        tempFeatures = list(currentFeatures)
        tempFeatures.append(x)
        printFeatureSet(tempFeatures,featureAcc[i],1)
        i+=1
    accuracy = max(featureAcc)
    tempFeatures = list(currentFeatures)
    tempFeatures.append(featNUsed.pop(featureAcc.index(accuracy)))
    if(accuracy < bestAcc):
        print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
    printFeatureSet(tempFeatures,accuracy,2)
    return (tempFeatures,accuracy)

def forwardSelection(data, numFeatures, records):
    featureFlags = [i for i in range(1, numFeatures+1)]
    currentFeatures = []
    bestFeatures = []
    bestWeakFeatures = []
    bestAcc = 0.00000
    bestAcc2 = 0.00000
    start = time.clock()
    for x in range(numFeatures):
        currentFeatures, accuracy = findBestFeatureFS(data, featureFlags, currentFeatures, bestAcc)
        if (accuracy > bestAcc):
            bestAcc = accuracy
            bestFeatures = list(currentFeatures)
        if len(currentFeatures) > (len(bestFeatures)):
            if (accuracy > bestAcc2):
                bestAcc2 = accuracy
                bestWeakFeatures = list(currentFeatures)
    end = time.clock()
    # print("Time Taken = ",end-start)
    if len(bestWeakFeatures) == len(bestFeatures):
        print("Feature Set {",','.join(str(j) for j in bestFeatures),"} was best, accuracy is ",bestAcc,"%")
    else:
        print("Feature Set {", ','.join(str(j) for j in bestFeatures), "} was best, accuracy is ", bestAcc, "%")
        print("Feature Set {", ','.join(str(j) for j in bestWeakFeatures), "} was best with a weak feature, accuracy is ", bestAcc2, "%")

def findBestFeatureBE(data, featureFlags, currentFeatures, bestAcc):
    featNUsed = list(currentFeatures)
    featureAcc = [0.00000] * len(featNUsed)
    i = 0
    accuracy = 0
    for x in featNUsed:
        dataFlags = createDataFlag(featureFlags, currentFeatures, x,2)
        featureAcc[i] = calcAccuracy(data,dataFlags)
        tempFeatures = list(currentFeatures)
        tempFeatures.remove(x)
        printFeatureSet(tempFeatures,featureAcc[i],1)
        i+=1
    accuracy = max(featureAcc)
    tempFeatures = list(currentFeatures)
    tempFeatures.remove(featNUsed.pop(featureAcc.index(accuracy)))
    if(accuracy < bestAcc):
        print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
    printFeatureSet(tempFeatures,accuracy,2)
    return (tempFeatures,accuracy)


def backwardElimination(data,numFeatures, records):
    featureFlags = [i for i in range(1, numFeatures+1)]
    currentFeatures = copy.deepcopy(featureFlags)
    bestFeatures = copy.deepcopy(featureFlags)
    bestWeakFeatures = []
    bestWeakFeaturesTemp = []
    bestAcc = 0.00000
    bestAcc2 = 0.00000
    bestAcc2Temp = 0.00000
    start = time.clock()
    for x in range(numFeatures-1):
        currentFeatures, accuracy = findBestFeatureBE(data, featureFlags, currentFeatures, bestAcc)
        if (accuracy > bestAcc):
            bestAcc2 = bestAcc
            bestWeakFeatures = bestFeatures
            bestAcc = accuracy
            bestFeatures = list(currentFeatures)
    end = time.clock()
    # print("Time Taken = ",end-start)
    if len(bestWeakFeatures) == len(bestFeatures):
        print("Feature Set {",','.join(str(j) for j in bestFeatures),"} was best, accuracy is ",bestAcc,"%")
    else:
        print("Feature Set {", ','.join(str(j) for j in bestFeatures), "} was best, accuracy is ", bestAcc, "%")
        print("Feature Set {", ','.join(str(j) for j in bestWeakFeatures), "} was best with a weak feature, accuracy is ", bestAcc2, "%")

def findBestFeatureGSpecial(data,featureFlags, currentFeatures,bestAcc):
    featNUsed = [i for i in featureFlags if i not in currentFeatures]
    featureAcc = [[0.00000,0] for i in range(len(featNUsed))]
    i = 0
    accuracy = 0
    leastMisses = -1
    for x in featNUsed:
        dataFlags = createDataFlag(featureFlags, currentFeatures, x, 1)
        t = calcAccuracyGSpecial(data,dataFlags,leastMisses)
        featureAcc[i][0],featureAcc[i][1] = t[0], t[1]
        if leastMisses == -1 or leastMisses > featureAcc[i][1]:
            leastMisses = featureAcc[i][1]
        tempFeatures = list(currentFeatures)
        tempFeatures.append(x)
        printFeatureSet(tempFeatures,featureAcc[i][0],1)
        i+=1
    accuracy = (max(featureAcc,key=lambda l:l[0]))[0]
    tempFeatures = list(currentFeatures)
    for i in range(len(featNUsed)):
        if featureAcc[i][0] == accuracy:
            tempFeatures.append(featNUsed.pop(i))
            break
    if(accuracy < bestAcc):
        print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
    printFeatureSet(tempFeatures,accuracy,2)
    return (tempFeatures,accuracy)

def specialGSelection(data,numFeatures, records):
    featureFlags = [i for i in range(1, numFeatures+1)]
    currentFeatures = []
    bestFeatures = []
    bestWeakFeatures = []
    bestAcc = 0.00000
    bestAcc2 = 0.00000
    threshold = 2
    start = time.clock()
    for x in range(numFeatures):
        currentFeatures, accuracy = findBestFeatureGSpecial(data, featureFlags, currentFeatures, bestAcc)
        if (accuracy > bestAcc):
            bestAcc = accuracy
            bestFeatures = list(currentFeatures)
        if len(currentFeatures) > (len(bestFeatures)):
            if (accuracy > bestAcc2):
                bestAcc2 = accuracy
                bestWeakFeatures = list(currentFeatures)
        if ((bestAcc-accuracy)*100/bestAcc > threshold ) and bestAcc!=accuracy:
            print("Accuracy difference has fallen below threshold of",threshold,"%. Terminating Algorithm\n")
            break
    end = time.clock()
    # print("Time taken =",end-start)
    if len(bestWeakFeatures) == len(bestFeatures):
        print("Feature Set {",','.join(str(j) for j in bestFeatures),"} was best, accuracy is ",bestAcc,"%")
    else:
        print("Feature Set {", ','.join(str(j) for j in bestFeatures), "} was best, accuracy is ", bestAcc, "%")
        print("Feature Set {", ','.join(str(j) for j in bestWeakFeatures), "} was best with a weak feature, accuracy is ", bestAcc2, "%")

def findDefaultRate(data, records):
    classDict = {}
    defaultRate = 0.0
    for  x in data:
        if not str(x[0]) in classDict:
            classDict[str(x[0])] = 1
        else:
            classDict[str(x[0])] = classDict.get(str(x[0])) + 1
    for x in iter(sorted(classDict.keys())):
        classDict[x] = (float(classDict[x])/records)*100.0
        if defaultRate < classDict[x]:
            defaultRate = classDict[x]
    print("Default Rate = ",defaultRate,"%\n")

if __name__ == "__main__":

    algo = ""
    print("Welcome to Gautham's Feature Selection Algorithm")
    while(1):
        try:
            file = input("Type in the name of the file to test:\t")
            open(file)
        except EnvironmentError :
            print("Invalid filename or ",file," could not be found")
        else:
            break

    print("Type the number of the algorithm you want to run.")
    print("1)\tForward Selection")
    print("2)\tBackward Selection")
    print("3)\tGautham's Special Algorithm \n")

    while(1):
        algo = input()
        if algo != "1" and algo != "2" and algo != "3":
            print("Invalid choice")
            print("Please enter a valid choice")
            print("1)\tForward Selection")
            print("2)\tBackward Selection")
            print("3)\tGautham's Special Algorithm\n")
        else:
            break
    data = extractData(file)
    records = len(data)
    numFeatures = len(data[0])-1
    print("This dataset has",numFeatures,"features (not including attributes), with",records,"instances")
    data = normalizeData(data, numFeatures, records)
    allFeatureKNN(data,numFeatures,records)
    findDefaultRate(data,records)
    print("Beginning search")
    if algo == "1":
        forwardSelection(data,numFeatures,records)
    elif algo == "2":
        backwardElimination(data,numFeatures,records)
    else:
        specialGSelection(data,numFeatures,records)
