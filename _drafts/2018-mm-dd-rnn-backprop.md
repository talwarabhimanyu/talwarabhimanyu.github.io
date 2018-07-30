---
layout: post
title: Build an RNN from scratch (with derivations!)
date: 2018-mm-dd
---
In this post I will show how to build a Recurrent Neural Network (RNN) from scratch. I will derive all the mathematical results we nede for this task, and more importantly I will show how to implement all that math efficiently in Python.

## Terminology
To process a sequence of length $$T$$, an RNN uses $$T$$ copies of a Basic Unit. In the figure below, I have shown two units of an RNN. The parameters used by each Basic Unit are "tied together". That is, the weight matrices $$W_h$$, $$W_e$$ and biases $$b_1$$ and $$b_2$$, are the same for each unit.

![RNN Diagram](/images/RNN Diagram.png)

**Inputs**
* $$x^{t}$$ is the $$t^{th}$$ element of an input sequence of length $$T$$
* Each $$x^{t}$$ is a vector of dimensions $$d \times 1$$

**Parameters**
* $$W_h$$ is a matrix of dimensions $$D_h \times D_h$$
* $$W_e$$ is a matrix of dimensions $$D_h \times d$$

**Interim Variables**
** $$h


##
