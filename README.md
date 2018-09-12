# Hallo-Welt
ToDo:  

大图很好(https://blog.csdn.net/linxid/article/details/79104130)  
from-import导入import后面的模块或者函数；import导入import后面的模块或者函数。二者都是要引用import后面的路径。  

从python2.6开始，with就成为默认关键字了。With是一个控制流语句，跟if for while try之类的是一类，with可以用来简化try-finally代码，看起来比try finally更清晰，所以说with用很优雅的方式处理上下文环境产生的异常。  

对于找工作解决实际问题而言：数据（代表着分布式系统领域）>特征（数据挖掘与自然语言处理与图像）>模型（机器学习）。  
机器学习和数据挖掘是两个相差的比较多的领域，机器学习比较偏向数学问题的推导，所以在顶会上的很多paper更看重idea，不是很看重实验是否来源于真实数据（有一些实验数据会自己构造。而数据挖掘说土点就是老子就是会feature~的领域。  
没有万能的算法，只有在一定使用环境中最优的算法。  

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

## Decision Tree  

## K-NearestNeighbor  
邻近分类算法是数据挖掘分类技术中最简单的方法之一。 说的是每个样本都可以用它最接近的k个邻居来代表。  
kNN算法的指导思想是“近朱者赤，近墨者黑”，由你的邻居来推断出你的类别。 

欧式距离：  
马氏距离：马氏距离能够缓解由于属性的线性组合带来的距离失真，是数据的协方差矩阵。  
曼哈顿距离：  
切比雪夫距离：  
闵氏距离：r取值为2时：曼哈顿距离；r取值为1时：欧式距离。   
平均距离：  
弦距离：  
测地距离：  

## SVM  
这些点，在几何空间中也表示向量，那么就把这些能够用来确定超平面的向量称为支持向量（直接支持超平面的生成），于是该算法就叫做支持向量机了。   
所以这个算法的好处就很明显了，任你训练样本如何庞大，我只需要找到支持向量就能很好的完成任务了，计算量就大大缩小了。更神奇的是它的分类效果相当好，在手写识别领域，其分类效果好到可以与精心设计训练的最优秀的神经网络相当，不仅计算量小，而且不需要复杂的定制，适用性强，也难怪它如此受欢迎，被很多人称为目前最好的分类算法了.  

## NN  


# Unsupervised Learning: Clustering
无监督学习显然难度要更大，在只有特征没有标签的训练数据集中，通过数据之间的内在联系和相似性将他们分成若干类。    
Google新闻按照内容结构的不同分成财经，娱乐，体育等不同的标签，这就是一种聚类。   
