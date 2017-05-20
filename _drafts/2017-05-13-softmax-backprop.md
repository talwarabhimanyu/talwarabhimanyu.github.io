---
layout: post
title: Backpropogating Softmax Layer of a Neural Network
date: 2017-05-13
---
In this post I attempt to describe the calculus involved in backpropogating gradients for the Softmax layer of a neural network. I will use a sample network with the following architecture (this is same as the toy neural-net trained in CS231n's Winter 2016 Session, [Assignment 1](http://cs231n.github.io/assignments2016/assignment1/)). This is a fully-connected network - the output of each node in _Layer i_ goes as input into each node in _Layer i+1_.

**Input Layer:** Each training example consists of four features. If we have five training examples, then the entire training set input can be represented by this matrix:

$$
X = \left( \begin{array}{ccc}
x_{1,1} & \ldots & \ x_{1,4}\\
x_{2,1} & \ldots & \ x_{2,4}\\
\vdots & \vdots & \vdots \\
x_{5,1} & \ldots & \ x_{5,4}\\
\end{array} \right)
$$

**Hidden Layer:** For the i<sup>th</sup> training example, the j<sup>th</sup> node in this hidden layer will first calculate the weighted sum of the four feature inputs and add some bias to the weighted sum, to calculate what is called Z1<sub>i,j</sub>. It will then apply the layer's activation function on Z1<sub>i,j</sub> to calculate the output O1<sub>ij</sub>. The whole operation will look like the following.

$$
\mathbf{Z1 = XW1 + b1}
$$

$$
\mathbf{O1 = _f_(Z1)}
$$

where, W1, the matrix of weights between Input and Hidden layers, is as follows:

$$
W1 = \left( \begin{array}{ccc}
w_{1,1} & \ldots & \ w_{1,10}\\
w_{2,1} & \ldots & \ w_{2,10}\\
\vdots & \vdots & \vdots \\
w_{4,1} & \ldots & \ w_{4,10}\\
\end{array} \right)
$$

and, b1, the relevant bias matrix is as follows:

$$
b1 = \left( \begin{array}{c}
b_{1} \\
\vdots \\
b_{10} \\
\end{array} \right)
$$

**Output Layer:** This is the Softmax layer. For i<sup>th</sup> training example, the k<sup>th</sup> node in this output layer will first calculate a _score_, which is a weighted sum of the ten inputs it receives from the Hidden layer.






