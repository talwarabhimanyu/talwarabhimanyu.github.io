---
layout: post
title: Backpropogating Softmax Layer of a Neural Network
date: 2017-05-13
---
In this post I would like to describe the calculus involved in backpropogating gradients for the Softmax layer of a neural network. I will use a sample network with the following architecture (this is same as the toy neural-net trained in CS231n's Winter 2016 Session, [Assignment 1](http://cs231n.github.io/assignments2016/assignment1/)). This is a fully-connected network - the output of each node in _Layer i_ goes as input into each node in _Layer i+1_.

**Input Layer:** Each training example consists of four features. If we have five training examples, then the entire training set input can be represented by this matrix:
$$
M = \left( \begin{array}{ccc}
x_{11} & \ldots & \ x_{14}\\
x_{21} & \ldots & \ x_{24}\\
\vdots & \vdots & \ldots \\
x_{51} & \ldots & \ x_{54}\\
\end{array} \right)
$$
**Hidden Layer:**
**Output Layer:** This is the Softmax layer.






