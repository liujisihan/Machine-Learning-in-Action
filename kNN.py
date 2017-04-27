#coding=utf-8
'''
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
'''
#使用sklearn实现
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors,datasets

#简单使用
# n_neighbors=15
# iris=datasets.load_iris()#加载数据集
# X=iris.data
# y=iris.target
# clf=neighbors.KNeighborsClassifier(n_neighbors)
# clf.fit(X,y)
# print X,clf.predict([[7.2,3.0,5,2],[5.2,3.5,1.5,0.2]])

#使用鸢尾花数据并作图
# n_neighbors=15
# iris=datasets.load_iris()#加载数据集
# X=iris.data[:,:2]#只用前两维
# y=iris.target
# h=.02
# cmap_light=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
# for weights in ["uniform","distance"]:
#     clf=neighbors.KNeighborsClassifier(n_neighbors,weights,algorithm='auto',leaf_size=30)
#     clf.fit(X,y)
#     x_min,x_max=X[:,0].min()-1,X[:,0].max()+1#第一维的最小值和最大值
#     y_min,y_max=X[:,1].min()-1,X[:,1].max()+1#第二维的最小值和最大值
#     xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
#     Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
#     Z=Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("3-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))
# plt.show()

#找到最近邻，方法1：
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# nbrs = neighbors.NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(X)#维度较低如低于20时用kd_tree，更低时用brute_force蛮力算法
# distances, indices = nbrs.kneighbors([[0,1]])#找到与(0,1)点最近的3个点,也可以把多个点放在一个列表里作为参数
# print indices,distances# distances是与(0,1)点最近的3个点的索引，distances是与这3个点的距离

#找到最近邻，方法2：
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# kdt=neighbors.KDTree(X,leaf_size=30,metric='euclidean')#Ball_tree用法相同，leaf_size越大，树的构建越快，但是查询越慢
# indices=kdt.query([[0,1]],return_distance=False)
# print indices

