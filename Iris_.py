#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : HDH_LIM
# @Site    : 
# @File    : decisiontree.py
# @Software: PyCharm
# 决策树演示
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
import os

# 导入路径
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# 导入数据
iris = load_iris()

# 构建模型
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# 保存模型
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

# 画图，保存到pdf文件
# 设置图像参数
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

# 保存图像到pdf文件
graph.write_pdf("iris.pdf")





