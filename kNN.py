#coding=utf-8
from numpy import *
import operator
from os import listdir
def CreateDataSet():#手动创建数据集，仅用于测试
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def Classify0(inX,dataSet,labels,k):#分类函数
    dataSetSize=dataSet.shape[0] # 每行一个数据
    diffMat=tile(inX,(dataSetSize,1))-dataSet # tile函数
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)#在列方向上求和
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#返回指定键的值，不存在该键返回0
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
#group,label=CreateDataSet()测试代码
#print Classify0([1,1.1],group,label,2)测试代码
def img2vector(filename):#将文本转换成向量
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def handWritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
        classifierResult=Classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifierResult,classNumStr)
        if (classifierResult!=classNumStr):errorCount+=1.0
    print 'the totalnumber of errors is: %d' % errorCount
    print '\nthe total error rate is: %f' % (errorCount/float(mTest))
handWritingClassTest()

#使用sklearn实现
'''
# from sklearn import neighbors
# knn=neighbors.KNeighborsClassifier()
# def get_data_and_labels(file_list):
#     training_files=listdir(file_list)
#     m=len(training_files)
#     labels=[]
#     data=zeros((m,1024))
#     for i in range(m):
#         file_name=training_files[i] #文件名
#         class_num=(file_name.split('.')[0]).split('_')[0] #类标签
#         labels.append(int(class_num))
#         data[i,:]=img2vector(file_list+'/%s' % file_name)
#     return data,array(labels)
# training_data,training_labels=get_data_and_labels('trainingDigits')
# print training_data.shape,training_labels.shape
# test_data,test_labels=get_data_and_labels('testDigits')
# print test_data.shape,test_labels.shape
# knn.fit(training_data,training_labels)
# error_count=0
# for i in range(test_data.shape[0]):
#     pre=knn.predict(test_data[i,:])
#     if pre!=test_labels[i]:
#         error_count+=1
# print error_count
'''
