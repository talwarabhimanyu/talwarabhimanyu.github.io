---
layout: post
title: Build an LSTM from scratch in Python (+ backprop derivations!)
date: 2018-08-12
tags: sequence-modeling lstm backprop-maths
---
## Introduction
In my [last post on Sequence Modelling](https://talwarabhimanyu.github.io/blog/2018/07/31/rnn-backprop), I derived the equations required for backpropogation through an RNN, and used those equations to implement [an RNN in Python](https://github.com/talwarabhimanyu/Learning-by-Coding/blob/master/Deep%20Learning%20from%20Scratch/RNN%20from%20Scratch/RNN%20from%20Scratch.ipynb) (without using PyTorch or Tensorflow). Through that post I demonstrated two tricks which make backprop through a network with 'tied up weights' easier to comprehend - use of 'dummy variables' and 'accumulation of gradients'. **In this post I intend to look at another neural network architecture known as an LSTM (Long Short-Term Memory), which builds upon RNNs, and manages to avoid the issue of vanishing gradients faced by RNNs.**

The mathematics used is not too dissimilar from what is required for RNNs (except that you will see a lot more alphabetds because there are a lot more parameters). That said, one has to be careful about the flow of 'influence' from various nodes in the network to the loss computation (if this ain't clear now, that's okay - it will become clearer during the course of our derivations below). This complication arises from the introduction of an extra 'internal state' variable (which was absent in RNNs), which is what will help us avoid vanishing gradients. It is due to this complication that I thought LSTMs deserve a new blog post. I will urge you to read my post on RNNs before proceeding because it introduces some key tricks which I will reuse for LSTMs.

I will assume that the reader is familiar with LSTMs. Chris Olah's [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) offers a very intuitive understanding of why LSTMs are structured the way they are.

## Terminology
To process a sequence of length $$T$$, an LSTM uses T copies of a Basic Unit. Each Unit uses the same set of parameters (weights and biases). E.g. say there is a 'root' version of a weight matrix $$W$$, then each LSTM Unit uses this same version, and any changes to the 'root' are reflected in the weights uses by each Unit. We say the parameters are 'tied together' across Units.

### Notation
I have tried to use the same alphabets to denote various parameters as used in Chapter 10 (Sequence Modelling) of the [_Deep Learning_](https://www.deeplearningbook.org/) book by Goodfellow, Bengio, and Courville. I will use the same notation as I used for my blog post on RNNs:

* Superscript $$(t)$$, such as in $$h^{(t)}$$ refers to value of a variable (in this case $$h$$) at time-step $$t$$.
* Subscript $$[i]$$ in square brackets, such as in $$h^{(t)}_{[i]}$$ refers to the $$i^{th}$$ component of the vector $$h^{(t)}$$.
* The symbols $$\times$$ and $$\circ$$ refer to scalar and element-wise multiplication respectively. In absence of a symbol, assume matrix multiplication.
* Superscript $$Tr$$, such as in $$W_h^{Tr}$$, implies Transpose of the matrix $$W_h$$.

Similar to the case of RNNs, I will break down the computation inside an LSTM into three parts:

### (1) LSTM Unit Computation
The LSTM Unit at time-step $$t$$ takes as inputs:
* $$x^{(t)}$$, a vector of dimensions $$d \times 1$$, which represents the $$t^{th}$$ 'word' in a sequence of length $$T$$, and
* $$h^{(t-1)}$$, a vector of dimensions $$D \times 1$$, which is the output of the previous LSTM Unit, and is referred to as a 'hidden-state' vector.
* $$s^{(t-1)}$$, a vector of dimensions $$D \times 1$$, which is the output of the previous LSTM Unit, and is referred to as an 'internal-state' vector.

_Note: The numbers $$D$$ and $$d$$ are hyperparameters._

A distinguishing feature of an LSTM vs. an RNN is the presence of three 'Gates' - $$g^{(t)}$$ (_input_), $$s^{(t)}$$ (_internal state_) and $$q^{(t)}$$ (_output_) - which are simply multiplicative factors (value of each $$\in \space [0,1]$$) which 'control' how much of the input $$x^{(t)}$$ and previous internal state $$s^{(t-1)}$$ can be used by the LSTM Unit at time-step $$t$$ for computation, and also how much of the hidden-state vector $$h^{(t)}$$ will be sent as output from this Unit. How this 'control' is exercised by the Gates will become clear from the following equations:

Let's first look at how the hidden-state $$h$$ and internal-state $$s$$ are computed at time-step $$t$$:

$$
\underbrace{h^{(t)}}_{\text{hidden state}} = \underbrace{q^{(t)}}_{\text{output gate}} \circ tanh(\underbrace{s^{(t)}}_{\text{internal state}})
$$

$$
s^{(t)} = \underbrace{f^{(t)}}_{\text{forget gate}} \circ s^{(t-1)} + \underbrace{g^{(t)}}_{\text{input gate}} \circ \underbrace{e^{(t)}}_{\text{input feature \n vector}}
$$

Now let's look at how the input feature vector $$e^{(t)}$$ and the three gates are computed:

$$
f^{(t)} = \sigma \left( b_{f} + U_{f}x^{(t)} + W_{f}h^{(t-1)} \right)
$$
