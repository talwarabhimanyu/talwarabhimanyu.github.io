---
layout: post
title: Build an LSTM from scratch in Python (+ backprop derivations!)
date: 2018-08-12
tags: sequence-modeling lstm backprop-maths
---
## TL;DR
In this blog post:
1. I derive equations for backpropogation-through-time for an LSTM.
2. I create an LSTM model in Python (without using Pytorch or Tensorflow).

## Introduction
In my [last post on Sequence Modelling](https://talwarabhimanyu.github.io/blog/2018/07/31/rnn-backprop), I derived the equations required for backpropogation through an RNN, and used those equations to implement [an RNN in Python](https://github.com/talwarabhimanyu/Learning-by-Coding/blob/master/Deep%20Learning%20from%20Scratch/RNN%20from%20Scratch/RNN%20from%20Scratch.ipynb) (without using PyTorch or Tensorflow). Through that post I demonstrated two tricks which make backprop through a network with 'tied up weights' easier to comprehend - use of 'dummy variables' and 'accumulation of gradients'. **In this post I intend to look at another neural network architecture known as an LSTM (Long Short-Term Memory), which builds upon RNNs, and overcomes the issue of vanishing gradients faced by RNNs.**

The mathematics used is not dissimilar from what is required for RNNs (although you will see a lot more alphabets and subscripts because there are a lot more parameters than in an RNN). That said, one has to be careful about the flow of 'influence' from various nodes in the network to the network's loss (I will explain this below). This need to be careful arises from the introduction of an 'internal state' variable in LSTMs. This 'internal state' is actually what will help us overcome the issue of vanishing gradients. 

It is due to this complication that I thought LSTMs deserve a new blog post. I will urge you to read my [post on RNNs](https://talwarabhimanyu.github.io/blog/2018/07/31/rnn-backprop) before proceeding because it introduces some key tricks which I will reuse for LSTMs. I will assume that the reader is familiar with LSTMs. Chris Olah's [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) offers a very intuitive explanation of why LSTMs are structured the way they are.

## Terminology
To process a sequence of length $$T$$, an LSTM uses $$T$$ copies of a Basic Unit. Each Unit uses the same set of parameters (weights and biases). E.g. say there is a 'root' version of a weight matrix $$W$$, then each LSTM Unit uses this same version, and any changes to the 'root' are reflected in the weights uses by each Unit. We say the parameters are 'tied together' across Units.

**Figure: Structure of an LSTM Network (showing a single LSTM Unit)**
Note:
* Variables in blue color are the parameters of the network. We need to learn these parameters through training on data.
* Scroll below for equations that govern the computation inside each of the four rectangular boxes in the diagram.


![LSTM Diagram](/images/LSTM Diagram.png)

### Notation
I have tried to use the same alphabets to denote various parameters as used in Chapter 10 (Sequence Modelling) of the [_Deep Learning_](https://www.deeplearningbook.org/) book by Goodfellow, Bengio, and Courville. I will use the same notation as I used for my blog post on RNNs:

* Superscript $$(t)$$, such as in $$h^{(t)}$$ refers to value of a variable (in this case $$h$$) at time-step $$t$$.
* Subscript $$[i]$$ in square brackets, such as in $$h^{(t)}_{[i]}$$ refers to the $$i^{th}$$ component of the vector $$h^{(t)}$$.
* The symbols $$\times$$ and $$\circ$$ refer to scalar and element-wise multiplication respectively. In absence of a symbol, assume matrix multiplication.
* Superscript $$Tr$$, such as in $$W_f^{Tr}$$, implies Transpose of the matrix $$W_f$$.

Similar to the case of RNNs, I will break down the computation inside an LSTM into three parts:
1. Computation inside the LSTM Units, which is what I will cover in detail in this blog post.
2. Computatoin at the Affine Layer, which takes the 'hidden-state' $$h^{(t)}$$ at each time-step $$t$$ as input, and applies an Affine transformation to produce a vector $$\theta^{(t)}$$ of length $$V$$ (the size of our Vocabulary). This layer is no different from the one I discussed in my blog post on RNNs and I will not discuss it further in this post.
3. Computation at the Softmax Layer, where at each time-step, the vetor $$\theta^{(t)}$$ is used to compute a probability distribution over our Vocabulary of $$V$$ words. Again, this is no different from what I discussed for RNNs, and I will not discuss it further in this post.  

### LSTM Unit Computation
The LSTM Unit at time-step $$t$$ takes as inputs:
* $$x^{(t)}$$, a vector of dimensions $$d \times 1$$, which represents the $$t^{th}$$ 'word' in a sequence of length $$T$$, and
* $$h^{(t-1)}$$, a vector of dimensions $$D \times 1$$, which is the output of the previous LSTM Unit, and is referred to as a 'hidden-state' vector.
* $$s^{(t-1)}$$, a vector of dimensions $$D \times 1$$, which is the output of the previous LSTM Unit, and is referred to as an 'internal-state' vector.

_Note: The numbers $$D$$ and $$d$$ are hyperparameters._

A feature which distinguishes LSTMs from RNNs is the presence of three 'Gates' - $$g^{(t)}$$ (_input_), $$s^{(t)}$$ (_internal state_) and $$q^{(t)}$$ (_output_). These are simply multiplicative factors (value of each $$\in \space [0,1]$$) which 'control' how much of the input $$x^{(t)}$$ and previous internal state $$s^{(t-1)}$$ can be used by the LSTM Unit at time-step $$t$$ for computation, and also how much of the hidden-state vector $$h^{(t)}$$ will be sent as output from this Unit. How this 'control' is exercised by the Gates will become clear from the following equations:

Let's first look at how the hidden-state $$h$$ and internal-state $$s$$ are computed at time-step $$t$$:

$$
\underbrace{h^{(t)}}_{\substack{\text{hidden} \\ \text{state}}} = \underbrace{q^{(t)}}_{\substack{\text{output} \\ \text{gate}}} \circ tanh(\underbrace{s^{(t)}}_{\substack{\text{internal} \\ \text{state}}})
$$

$$
s^{(t)} = \underbrace{f^{(t)}}_{\substack{\text{forget} \\ \text{gate}}} \circ s^{(t-1)} + \underbrace{g^{(t)}}_{\substack{\text{input} \\ \text{gate}}} \circ \underbrace{e^{(t)}}_{\substack{\text{input} \\ \text{feature} \\ \text{vector}}}
$$

Now let's look at how the input feature vector $$e^{(t)}$$ and the three gates are computed:

$$
\begin{align}
\text{(Forget Gate) } f^{(t)} &= \sigma \left( b_{f} + U_{f}x^{(t)} + W_{f}h^{(t-1)} \right)
\\
\text{(Input Gate) } g^{(t)} &= \sigma \left( b_{g} + U_{g}x^{(t)} + W_{g}h^{(t-1)} \right)
\\
\text{(Output Gate) } q^{(t)} &= \sigma \left( b_{q} + U_{q}x^{(t)} + W_{q}h^{(t-1)} \right)
\\
\text{(Input Feature Vector) } e^{(t)} &= \sigma \left( b_{e} + U_{e}x^{(t)} + W_{e}h^{(t-1)} \right)
\end{align}
$$

The $$tanh$$ function, also known as the Hyperbolic Tangent, is defined as follows:

$$
tanh(x) = \frac {1 \space - \space e^{-2x}} {1 \space + \space e^{-2x}}
$$

Its derivative with respect to $$x$$ at a point, can be computed in terms of the value of $$tanh$$ at that point:

$$
tanh'(x) = (1 \space - \space tanh^{2}(x))
$$

