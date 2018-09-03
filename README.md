# Hallo-Welt
Test how to use Github
Sounds it is a very cool staaf to collaborate with other `Geeks` in the world.

李飞飞 cs231n

1. [Assignment translated into chinese](http://op.inews.qq.com/m/20180207A0GNWA00?refer=100000355&chl_code=kb_news_tech&h=0)  
2. [CS231n官方笔记授权翻译总集篇发布](https://zhuanlan.zhihu.com/p/21930884)

# Supervised Learning: Classification and Regression 
分类和回归的区别在于输出变量的类型。  
定量输出称为回归，或者说是连续变量预测。找到最有拟合。  
定性输出称为分类，或者说是离散变量预测。寻找决策边界。  
分类和回归从某种意义上又像是一种反义词，比如从字面上理解分和归，比如从图上理解，分类是找一条线或者超平面去分开数据，而回归则相反找一条线去尽可能的拟合逼近这些数据。这应该是最直译的。  
举个例子：  
预测明天的气温是多少度，这是一个回归任务。  
预测明天是阴、晴还是雨，就是一个分类任务。  
若我们欲预测的是离散值，例如"好瓜""坏瓜"，此类学习任务称为 "分类"。  
若欲预测的是连续值，例如西瓜的成熟度0.95 ,0.37,此类学习任务称为"回归"。  

# SVM  
这些点，在几何空间中也表示向量，那么就把这些能够用来确定超平面的向量称为支持向量（直接支持超平面的生成），于是该算法就叫做支持向量机了。   
所以这个算法的好处就很明显了，任你训练样本如何庞大，我只需要找到支持向量就能很好的完成任务了，计算量就大大缩小了。更神奇的是它的分类效果相当好，在手写识别领域，其分类效果好到可以与精心设计训练的最优秀的神经网络相当，不仅计算量小，而且不需要复杂的定制，适用性强，也难怪它如此受欢迎，被很多人称为目前最好的分类算法了.  
