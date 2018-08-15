---
layout: post
title: Build an LSTM from scratch in Python (+ backprop derivations!)
date: 2018-08-12
tags: sequence-modeling lstm backprop-maths
---
## TL;DR
In this blog post:
1. I derive equations for backpropogation-through-time for an LSTM.
2. I illustrate how NOT considering all paths of influence flow in an LSTM, can botch up our chain-rule application. 
3. I create an LSTM model in Python (without using Pytorch or Tensorflow).

## Introduction
In my [last post on Sequence Modelling](https://talwarabhimanyu.github.io/blog/2018/07/31/rnn-backprop), I derived the equations required for backpropogation through an RNN, and used those equations to implement [an RNN in Python](https://github.com/talwarabhimanyu/Learning-by-Coding/blob/master/Deep%20Learning%20from%20Scratch/RNN%20from%20Scratch/RNN%20from%20Scratch.ipynb) (without using PyTorch or Tensorflow). Through that post I demonstrated two tricks which make backprop through a network with 'tied up weights' easier to comprehend - use of 'dummy variables' and 'accumulation of gradients'. **In this post I intend to look at another neural network architecture known as an LSTM (Long Short-Term Memory), which builds upon RNNs, and overcomes the issue of vanishing gradients faced by RNNs.**

The mathematics used is not dissimilar from what is required for RNNs (although you will see a lot more alphabets and subscripts because there are a lot more parameters than in an RNN). That said, one has to be careful about the flow of 'influence' from various nodes in the network to the network's loss (I will explain this below). It is easy to miss out on a 'path' in the network through which 'influence' flows (I speak from personal experience!) 

I strongly suggest you read my [post on RNNs](https://talwarabhimanyu.github.io/blog/2018/07/31/rnn-backprop) before proceeding because it introduces some key tricks which I will reuse in this post. I will assume that the reader is familiar with LSTMs. Chris Olah's [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) offers a very intuitive explanation of why LSTMs are structured the way they are.

## Terminology
To process a sequence of length $$T$$, an LSTM uses $$T$$ copies of a Basic Unit (henceforth referred to as just a Unit). Each Unit uses the same set of parameters (weights and biases). One way to understand this is that there is a 'root' version of a weight matrix $$W$$, and each Unit uses this same version. Any changes to the 'root' are reflected in the matrix $$W$$ used by each Unit. We sometimes say that the parameters are 'tied together' across Units.

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
1. **LSTM Units:** I will cover this in detail in this post.
2. **Affine Layer:** This layer applies an Affine Transformation to the 'hidden-state' $$h^{(t)}$$ at each time-step $$t$$, to produce a vector $$\theta^{(t)}$$ of length $$V$$ (the size of our Vocabulary). The formula below describes the computation ($$U$$ and $$b$$ are model parameters):

$$
\theta^{(t)} = Uh^{(t)} + b
$$

3. **Softmax Layer:** At each time-step, the vector $$\theta^{(t)}$$ is used to compute a probability distribution over our Vocabulary of $$V$$ words. The formula below describes this:

$$
\hat{y}^{(t)}_{[i]} = \frac {e^{\theta^{(t)}_{[i]}}} {\sum_{j=0}^{V-1} e^{\theta^{(t)}_{[i]}}} \tag{3.1}
$$

I have discussed the Affine and Softmax computations in my blog post on RNNs. The same discussion holds for LSTMs as well and so I will not be talking about those layers here. 

The loss attributed to this time-step, $$J^{(t)}$$ is given by:

$$
J^{(t)} = -\sum_{i=0}^{V-1} y^{(t)}_{[i]} log \hat{y}^{(t)}_{[i]} \tag{3.2}
$$

The vector $$y^{(t)}$$ is a one-hot vector with the same dimensions as that of $$\hat{y}^{t}$$ - it contains a $$1$$ at the index of the 'true' next-word for time-step $$t$$. And finally, the overall loss for our LSTM is the sum of losses contributed by each time-step:

$$
J = \sum_{t=1}^{T} J^{(t)} \tag{3.3}
$$

**GOAL:** We want to find the gradient of $$J$$ with respect to each and every element of parameter matrices and vectors. For the sake of length of this post, I will only demonstrate all the maths required to calculate gradients w.r.t $$W_f$$, but I believe that after reading this, you will be able to apply the same concepts for other parameters. You can always refer to my [Jupyter Notebook](https://github.com/talwarabhimanyu/Learning-by-Coding/blob/master/Deep%20Learning%20from%20Scratch/LSTM%20from%20Scratch/LSTM%20from%20Scratch.ipynb) to understand how rest of the gradients should be calculated.

### LSTM Unit Computation
The LSTM Unit at time-step $$k$$ takes as inputs:
* $$x^{(k)}$$, a vector of dimensions $$d \times 1$$, which represents the $$t^{th}$$ 'word' in a sequence of length $$T$$, and
* $$h^{(k-1)}$$, a vector of dimensions $$D \times 1$$, which is the output of the previous LSTM Unit, and is referred to as a 'hidden-state' vector.
* $$s^{(k-1)}$$, a vector of dimensions $$D \times 1$$, which is the output of the previous LSTM Unit, and is referred to as an 'internal-state' vector.

_Note: The numbers $$D$$ and $$d$$ are hyperparameters._

**A feature which distinguishes LSTMs from RNNs is the presence of three 'Gates' - $$g^{(k)}$$ (_input_), $$f^{(k)}$$ (_forget_) and $$q^{(k)}$$ (_output_).** The value of each of these Gates lies in the range $$[0, \space 1]$$, and it determines how much the input $$x^{(k)}$$, the previous internal-state $$s^{(k-1)}$$ and the next internal-state $$s^{(k)}$$ will contribute towards our computation. **These Gates act like knobs, where 0 implies no contribution from the variable controlled by a knob, and 1 implies full contribution.** The equations below will make this clearer.

Let's look at how the hidden-state $$h$$ and internal-state $$s$$ are computed at time-step $$t$$:

$$
\underbrace{h^{(t)}}_{\substack{\text{hidden} \\ \text{state}}} = \underbrace{q^{(t)}}_{\substack{\text{output} \\ \text{gate}}} \circ tanh(\underbrace{s^{(t)}}_{\substack{\text{internal} \\ \text{state}}}) \tag{1.1}
$$

$$
s^{(t)} = \underbrace{f^{(t)}}_{\substack{\text{forget} \\ \text{gate}}} \circ s^{(t-1)} + \underbrace{g^{(t)}}_{\substack{\text{input} \\ \text{gate}}} \circ \underbrace{e^{(t)}}_{\substack{\text{input} \\ \text{feature} \\ \text{vector}}} \tag{1.2}
$$

Now let's look at how the input feature vector $$e^{(t)}$$ and the three gates are computed:

$$
\begin{align}
\text{(Forget Gate) } f^{(t)} &= \sigma \underbrace{\left( b_{f} + U_{f}x^{(t)} + W_{f}h^{(t-1)} \right)}_{z_f} \tag{2.1}
\\
\text{(Input Gate) } g^{(t)} &= \sigma \underbrace{\left( b_{g} + U_{g}x^{(t)} + W_{g}h^{(t-1)} \right)}_{z_g} \tag{2.2}
\\
\text{(Output Gate) } q^{(t)} &= \sigma \underbrace{\left( b_{q} + U_{q}x^{(t)} + W_{q}h^{(t-1)} \right)}_{z_q} \tag{2.3}
\\
\text{(Input Feature Vector) } e^{(t)} &= \sigma \underbrace{\left( b_{e} + U_{e}x^{(t)} + W_{e}h^{(t-1)} \right)}_{z_e} \tag{2.4}
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

## Applying the Two BPTT Tricks
We will make use of the two Backpropogation-Through-Time (BPTT) tricks that I had described in detail in my blog post on RNNs. Using the 'Dummy Variables' trick, we will pretend that the LSTM Unit at time-step $$k$$ has its own copy of parameters. For example instead of the matrices $$W_f$$ and $$U_f$$, we will pretend that this Unit uses parameters $$W_f^{(k)}$$ and $$U_f^{(k)}$$. We can say:

$$
\begin{align}
\frac {\partial J^{(t)}} {\partial W_{f [i,j]}} &=  \sum_{k=1}^{T} \frac {\partial J^{(t)}} {\partial W_{f [i,j]}^{(k)}} \times \underbrace{\frac {\partial W_{f [i,j]}^{(k)}} {\partial W_{f[i,j]}}}_{Equals \space 1.} \\
\\
&=  \sum_{k=1}^{T} \frac {\partial J^{(t)}} {\partial W_{f [i,j]}^{(k)}} \tag{xx}\\
\end{align}
$$

## Blueprint for Computing Gradients
We ask ourselves the same big picture questions which we had asked for RNNs:
1. What information do we need at each time-step to compute gradients?
2. How do we pass that information efficiently between layers?
We will find answers to these below.

### Tracing Paths of Influence
Observe that our parameter of interest, $$W_f$$ appears only in one of the equations, $$Eq. 2.1$$. Focussing on the version of $$W_f$$ used by the Unit at time-step $$k$$, i.e. $$W_f^{(k)}$$, we can see that a change in the value of $$W_f^{(k)}$$ will cause a change in the value of $$f^{(k)}$$, which in turn will impact the value of loss for our LSTM. 

**We are basically tracing paths through which 'influence' could flow from our variable of interest to the value of loss for our LSTM! In this case, we've discovered that there is a path from $$W_f^{(k)}$$ to the loss quantity $$J^{(t)}$$, via $$f^{(k)}$$. Moreover, we have observed that there is NO PATH between $$W_f^{(k)}$$ and $$J^{(t)}$$ which avoids $$f^{(k)}$$.** Therefore, if we know the gradient of loss w.r.t. $$f^{(k)}$$, we can just restrict our task to understanding how 'influence' flows from $$W_f^{(k)}$$ to $$f^{(k)}$$, and we should be able to compute the gradient of loss w.r.t. $$W_f^{(k)}$$.

Utilizing our knowledge of this one-and-only path (and using chain-rule to 'traverse' this path), we can now say:

$$
\begin{align}
\frac {\partial J^{(t)}} {\partial W_{f[i,j]}^{(k)}} &= \sum_{p=1}^{D} \underbrace{\frac {\partial J^{(t)}} {\partial f_{[p]}^{(k)}}}_{\text{Eq. xx1}} \times \underbrace{\frac {\partial f_{[p]}^{(k)}} {\partial W_{f[i,j]}^{(k)}} }_{\text{Eq. xx2}}  \tag{xx}
\\
\end{align}
$$

The second quantity in this expression is straighforward to compute using $$Eq. 2.1$$:

$$
\begin{align}
\frac {\partial f_{[p]}^{(k)}} {\partial W_{f[i,j]}^{(k)}} &=
\begin{cases}
0, & \text{p $\ne$ i} \\[2ex]
\sigma'(z_{f[i]}^{(k)}) \times h_{[j]}^{(k-1)}, & \text{p = i} \tag{xx2}
\end{cases}
\end{align}
$$

To compute the first quantity on the right-hand-side of $$Eq. xx$$, we trace paths of influence from $$f^{(k)}$$ towards loss $$J^{(t)}$$, and observe from $$Eq. 1.2$$ that such a path MUST pass through $$s^{(k)}$$. We use this insight to say:

$$
\begin{align}
\frac {\partial J^{(t)}} {\partial f_{[p]}^{(k)}} &= \sum_{m=1}^{D} \frac {\partial J^{(t)}} {\partial s_{[m]}^{(k)}} \times \frac {\partial s_{[m]}^{(k)}} {\partial f_{[p]}^{(k)}} \\[2ex]
&= \frac {\partial J^{(t)}} {\partial s_{[p]}^{(k)}} \times s_{[p]}^{(k-1)} \tag{xx1} 
\end{align}
$$

Substitute $$Eq. xx1$$ and $$Eq. xx2$$ in $$Eq. xx$$ to get:

$$
\begin{align}
\frac {\partial J^{(t)}} {\partial W_{f[i,j]}^{(k)}} &= \frac {\partial J^{(t)}} {\partial s_{[i]}^{(k)}} \times \sigma'(z_{f[i]}^{(k)}) \times s_{[i]}^{(k-1)} \times h_{[j]}^{(k-1)}
\end{align}
$$

This can be expressed in matrix notation as follows:

$$ \bbox[yellow,5px,border:2px solid red]
{
\frac {\partial J^{(t)}} {\partial W_{f}^{(k)}} = \left( \underbrace{\frac {\partial J^{(t)}} {\partial s^{(k)}}}_{\delta_{t}^{k}} \circ \underbrace{\sigma'(z_{f}^{(k)})}_{\text{Local}} \circ \underbrace{s^{(k-1)}}_{\text{Local}} \right) \underbrace{h^{(k-1)}_{\text{Local}} \space Tr}
\qquad (yy)
}
$$

We have now kind of answered the first big picture question - from $$Eq. yy$$, we can tell which quantities are required at time-step $$k$$ to compute gradients of loss w.r.t $$W_f^{(k)}$$. All quantities marked as 'Local' in the expression are available from the cache stored for time-step $$k$$ during the forward pass.

