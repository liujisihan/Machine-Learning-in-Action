#coding=utf-8
from math import log
import operator
def calcShannonEnt(dataSet):#计算香农熵
    numEntries=len(dataSet)#总的实例数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]#最后一个数是标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt
def create_data_set():#手动创建数据集，仅用于测试，在命令行下进入本目录，import trees即可使用
    dataSet=[[1,1,'yes'],
             [1,1,'yes'],
             [1,0,'no'],
             [0,1,'no'],
             [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels
def splitDataSet(data,axis,value):#按某个特征划分数据集，数据集是一个嵌套列表
    '在某个维度上以某个值划分数据集，例如用上面手动创建的数据集，在0维上以1划分，结果是[[1,"yes"],[1,"yes"],[0,"no"]]'
    retDataSet=[]#保存划分后的数据集
    for featVec in data:
        if featVec[axis] == value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet#划分后的数据将不包含该维度
def chooseBestFeatureToSplit(data):#选择最好的数据集划分方式，data是二维列表
    numFeatures=len(data[0])-1#特征数量
    baseEntropy=calcShannonEnt(data)#数据集基本香农熵
    bestInfoGain=0.0
    bestFeature=-1#最佳划分特征的维度
    for i in range(numFeatures):
        featList=[example[i] for example in data]#将所有数据的第i维取出
        uniqueVals=set(featList)#第i维的所有取值
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(data,i,value)#在第i维上以value划分
            prob=len(subDataSet)/float(len(data))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if (infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def createTree(dataSet,labels):#创建树
    classList = [example[-1] for example in dataSet]#取出类别标签
    if classList.count(classList[0])==len(classList):
        return classList[0]#类别完全相同，停止划分
    if len(dataSet[0])==1:#遍历完所有特征，只剩下类别标签，返回占多数的类别标签作为该叶节点的类别
        return majorityCnt(classList)#?
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]#?
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueValues=set(featValues)
    for value in uniqueValues:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
#至此，可以构建决策树，书上后续的代码是实现可视化
data,labels=create_data_set()
tree=createTree(data,labels)
print tree