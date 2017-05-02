#coding=utf-8

'''
from numpy import *
def loadDataSet():#创建数据集
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]#1代表侮辱性文字，0代表正常言论
    return postingList,classVec
def createVocabList(dataSet):#创建包含文档中所有词的不重复的列表
    vocabSet=set([])
    for document in  dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)
def setOfWords2Vec(vocabList,inputSet):#输入是词汇表和文档，返回一个01向量，表示词汇表中的单词是否在文档中出现
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print("the word: %s is not in my Vocabulary!" % word)
    return returnVec
def trainNB0(trainMatrix,trainCategory):#trainMatrix是经setOfWords2Vec()循环产生的每篇训练文档的01向量，是一个二维矩阵
    numTrainDocs=len(trainMatrix)#训练文档总数
    numWords=len(trainMatrix[0])#词汇表中的总词数
    pAbusive=sum(trainCategory)/float(numTrainDocs)#侮辱性文档概率
    p0Num=ones(numWords)#0类中各个词出现的次数，是一个向量
    p1Num=ones(numWords)#1类中...
    p0Denom=2.0#0类中词条总数
    p1Denom=2.0#1类中...
    for i in range(numTrainDocs):#对每篇训练文档
        if trainCategory[i]==1:#如果是1类
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)#1类中每个词出现的概率
    p0Vect=log(p0Num/p0Denom)#0类中...
    return p0Vect,p1Vect,pAbusive
def classfyNB(vec2Classify,p0Vec,p1Vec,pClass1):#vec2Classify是待分类向量
    p1=sum(vec2Classify*p1Vec)+log(pClass1)#log相加等于元素相乘，vec2Classify*p1Vec就是待分类向量每个分量的概率的积，P1是贝叶斯公式的分子部分
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)#同上
    if p1>p0:
        return 1
    else:
        return 0
def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classfyNB(thisDoc,p0V,p1V,pAb)
    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as: ',classfyNB(thisDoc,p0V,p1V,pAb)
'''

#用sklearn实现
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

train_doc=['my dog has flea problems help please',
                 'maybe not take him to dog park stupid',
                 'my dalmation is so cute I love him',
                 'stop posting stupid worthless garbage',
                 'mr licks ate my steak how to stop him',
                 'quit buying worthless dog food stupid']
y_train=[0,1,0,1,0,1]
test_doc=['my stupid dog is garbage','please help my dog','buying dog food is worthless']
y_test=[1,0,1]
vectorizer=CountVectorizer()

x_train=vectorizer.fit_transform(train_doc)#把训练数据转换成01向量，行数是训练文档数，列数是不重复的总词数
x_test=vectorizer.transform(test_doc)#把测试数据转换成01向量
clf=MultinomialNB().fit(x_train,y_train)#多项式模型
predicted=clf.predict(x_test)

precision,recall,thresholds=precision_recall_curve(y_test,predicted)
print("precision:",precision,"recall:",recall)
answer=clf.predict_proba(x_test)[:,1]
report=answer>0.5
print(classification_report(y_test,report,target_names=['pos','neg']))

