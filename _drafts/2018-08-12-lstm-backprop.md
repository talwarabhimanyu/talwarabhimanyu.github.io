---
layout: post
title: Build an LSTM from scratch (with derivations!)
date: 2018-08-12
tags: sequence-modeling lstm backprop-maths
---
## Introduction
In my [last post on Sequence Modelling](https://talwarabhimanyu.github.io/blog/2018/07/31/rnn-backprop), I derived the equations required for backpropogation through an RNN, and used those equations to implement [an RNN in Python](https://github.com/talwarabhimanyu/Learning-by-Coding/blob/master/Deep%20Learning%20from%20Scratch/RNN%20from%20Scratch/RNN%20from%20Scratch.ipynb) (without using PyTorch or Tensorflow). Through that post I demonstrated two tricks which make backprop through a network with 'tied up weights' easier to comprehend - use of 'dummy variables' and 'accumulation of gradients'. **In this post I intend to look at another neural network architecture known as an LSTM (Long Short-Term Memory), which builds upon RNNs, and manages to avoid the issue of vanishing gradients faced by RNNs.**

The mathematics used is not too dissimilar from what is required for RNNs (except that you will see a lot more alphabetds because there are a lot more parameters). That said, one has to be careful about the flow of 'influence' from various nodes in the network to the loss computation (if this ain't clear now, that's okay - it will become clearer during the course of our derivations below). This complication arises from the introduction of an extra 'internal state' variable (which was absent in RNNs), which is what will help us avoid vanishing gradients. It is due to this complication that I thought LSTMs deserve a new blog post. I will urge you to read my post on RNNs before proceeding because it introduces some key tricks which I will reuse for LSTMs. 
