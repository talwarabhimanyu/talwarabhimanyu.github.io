---
layout: post
title: Build an RNN from scratch (with derivations!)
date: 2018-mm-dd
---
In this post I will show how to build a Recurrent Neural Network (RNN) from scratch. I will derive all the mathematical results we need for this task, and more importantly I will show how to implement all that math efficiently in Python.

## Terminology
To process a sequence of length $$T$$, an RNN uses $$T$$ copies of a Basic Unit (henceforth referred to as just a Unit). In the figure below, I have shown two Units of an RNN. The parameters used by each Unit are "tied together". That is, the weight matrices $$W_h$$, $$W_e$$ and biases $$b_1$$ and $$b_2$$, are the same for each Unit. Each Unit is also referred to as a "time step".

![RNN Diagram](/images/RNN Diagram.png)

**Inputs**
* $$x^{t}$$ is the $$t^{th}$$ element of an input sequence of length $$T$$
* Each $$x^{t}$$ is a vector of dimensions $$d \times 1$$

**Parameters**
* $$W_h$$ is a matrix of dimensions $$D_h \times D_h$$
* $$b_1$$ is a bias vector of dimensions $$D_h \times 1$$
* $$W_e$$ is a matrix of dimensions $$D_h \times d$$
* $$U$$ is a matrix of dimensions $$\lvert V \rvert \times D_h$$ where $$V$$ is our 'Vocabulary', the set of objects (such as 'words') from which each element of a sequence is drawn. 
* $$b_2$$ is a bias vector of dimensions $$\lvert V \rvert \times 1$$

**Interim Variables**
* $$z^{t}$$ is a vector of dimensions $$D_h \times 1$$. It is computed as follows:
$$z^t = W_h h^{t-1} + W_x x^{t} + b_1$$

* $$h^t$$ is a vector of dimensions $$D_h \times 1$$. It is computer as follows:
$$h^t = \sigma(z^t)$$

where $$\sigma()$$ refers to the Sigmoid function, defined as:
$$\sigma(x) = \frac{1}{(1 + e^{-x})}$$

**Outputs**
* $$y^t$$ is a vector of dimensions $$\lvert V \rvert \times 1$$.

##
