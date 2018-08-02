---
layout: post
title: Build an RNN from scratch (with derivations!)
date: 2018-07-31
---
In this post I will (1) delve into the maths behins backpropogation through a Recurrent Neural Network (RNN), and (2) using the equations I derive, build an RNN in Python from scratch. I will assume that the reader is familiar with an RNN's structure and why they have become popular (this excellent [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) explains key ideas). All layers of an RNN use the same set of parameters (weights and biases are "tied together") - this is unlike a plain feedforward network where each layer has its own set of parameters. This aspect makes understanding backpropogation through an RNN a bit tricky. 

Several other resources on the web have tackled the maths behind an RNN, however I have found them lacking in detail on how exactly gradients are "accumulated" during backprop to deal with "tied weights". Therefore, I will attempt to explain that aspect in a lot of detail in this post.

## Terminology
To process a sequence of length $$T$$, an RNN uses $$T$$ copies of a Basic Unit (henceforth referred to as just a Unit). In the figure below, I have shown two Units of an RNN. The parameters used by each Unit are "tied together". That is, the weight matrices $$W_h$$, $$W_e$$ and biases $$b_1$$ and $$b_2$$, are the same for each Unit. Each Unit is also referred to as a "time step".

![RNN Diagram](/images/RNN Diagram.png)

The parameters used by this RNN are the weight matrices $$W_h$$, $$W_e$$, and $$U$$, and the bias vectors $$b_1$$ and $$b_2$$. During backprop, we need to calculated gradients of the training loss with respect to all of these parameters.

### RNN Unit Computation
The RNN Unit at time-step $$t$$ takes as inputs:
* $$x^{(t)}$$, a vector of dimensions $$d \times 1$$, which represents the $$t^th$$ 'word' in a sequence of length $$T$$, and
* $$h^{(t-1)}, a vector of dimensions $$D_h \times 1$$, which is the output of the previous RNN Unit, and is referred to as a 'hidden-state' vector.

The output of the RNN unit at time-step $$t$$ is its 'hidden-state vector' $$h^{(t)}$$. The equations governing a single unit are:
$$z^{(t)} = W_h h^{(t-1)} + W_x x^{(t)} + b_1$$ \tag{1.1}
$$h^{(t)} = \sigma(z^{(t)})$$ \tag{1.2}

where $$\sigma()$$ refers to the Sigmoid function, defined as:
$$\sigma(x) = \frac{1}{(1 + e^{-x})}$$

An RNN comprises a sequence of a number of such single RNN Units. **It is evident from these equations that a perturbation to the weight matrix $$W_h$$ will impact the value of a hidden-state vector $$h^{(t)}$$ not just directly via its presence in $$Eq. 1.1$$, but also indirectly via its impact on all hidden-state vectors $$h^{[1:t-1]}$$.** This aspect of an RNN makes the gradient calculations seem tricky but we will see two clever work-arounds to tackle this.

### Affine Layer
The hidden-state vector $$h^{(t)}$$ of RNN Unit at time-step $$t$$ is fed into (1) the next RNN Unit, and (2) through an Affine Layer which produces the vector $$\theta^{(t)}$$ of dimensions $$V \times 1$$, where $$V$$ is the size of our Vocabulary (set of all 'words' in our training-set if you are passing a word vector as input $$x^{(t)}$$ at time-step $$t$$, or a set of all characters in our training set if we are working on a character level RNN Model). The equations governing this layer are:

$$\theta^{(t)} = Uh^{(t)} + b_2$$

**Parameters**
* $$W_h$$ is a matrix of dimensions $$D_h \times D_h$$
* $$b_1$$ is a bias vector of dimensions $$D_h \times 1$$
* $$W_e$$ is a matrix of dimensions $$D_h \times d$$
* $$U$$ is a matrix of dimensions $$\lvert V \rvert \times D_h$$ where $$V$$ is our 'Vocabulary', the set of objects (such as 'words') from which each element of a sequence is drawn. 
* $$b_2$$ is a bias vector of dimensions $$\lvert V \rvert \times 1$$

**Interim Variables**
* $$z^{t}$$ is a vector of dimensions $$D_h \times 1$$. It is computed as follows:

* $$h^t$$ is a vector of dimensions $$D_h \times 1$$. It is computer as follows:
**Outputs**
* $$\hat{y}^{t}$$ is a vector of dimensions $$\lvert V \rvert \times 1$$. The $$i^{th}$$ element of this vector represents the probability with which the output at time-step $$t$$ is equal to the word located at the $$i^{th}$$ index in Vocabulary $$V$$. This vector is computed as follows:
$$\hat{y}^{t} = Softmax(Uh^{t} + b_2)$$

**The Loss Function**
At time-step $$t$$, we have a probability vector $$\hat{y}^{t}$$ which is the output of our model for this time-step. We also have with us the truth vector $$y^t$$ which is a one-hot vector with the same dimensions as that of $$\hat{y}^{t}$$.

##
