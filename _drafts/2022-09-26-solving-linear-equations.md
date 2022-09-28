---
layout: post
title: Linear Equations, once and for all
date: 2022-09-26
tags: linear-algebra
---
Yes, this is yet another blogpost on the internet about solving a system of linear equations. My aim here is to capture all possible cases that arise in this problem (skinny, fat, or square matrices; full rank or not; approximate or exact solution; no, unique or many solutions), so that when in doubt (about linear equations), I can revisit this post and get my answers. 

Here are my references, to which I credit everything in this post (except the errors, which are solely on me):

1. Prof. Gilbert Strang's course on [Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
2. Prof. Stephen Boyd's course [Introduction to Linear Dynamical Systems](https://ee263.stanford.edu/archive/)

## Introduction
We have unknowns $$x \in \mathbb{R}^n$$ and observations $$y \in \mathbb{R}^m$$. We _know_ that these are related as $$y = Ax$$ for a given $$A \in \mathbb{R}^{m \times n}$$. We want to know the value of $$x$$. Let $$\mathcal{C}(A), \mathcal{R}(A), \mathcal{N}(A)$$ denote the column, row, null spaces respectively. Three cases arise depending on the shape of $$A$$:

## $$A$$ is square ($$m=n$$)
If $$A$$ is full-rank, i.e. it is invertible, the solution is unique, $$x = A^{-1}y$$. If however, $$A$$ is not invertible, we have two cases:

1. $$y \notin \mathcal{C}(A)$$, in which case there is no solution (because no $$x$$ exists such that $$y = Ax$$). If $$A = \begin{bmatrix} 1 & 1 \\ 2 & 2 \end{bmatrix}$$, then $$\mathcal{C}(A)$$ only contains vectors of the form $$\begin{bmatrix} k \\ 2k\end{bmatrix}$$ and so $$y$$ will have to be of this form, for a solution to exist.

2. $$y \in \mathcal{A}(C)$$, in which case there exist multiple solutions. As $$A$$ is not full-rank, $$\mathcal{N}(A) \neq \Phi$$, and so if $$x_p$$ is some solution to this system, then $$x_p + x_n, x_n \in \mathcal{N}(A)$$ is also a solution (because $$y = Ax_p = Ax_p + 0 = Ax_p + Ax_n = A(x_n + x_p)$$). Taking the sane example as in 1. above, if $$A = \begin{bmatrix} 1 & 1 \\ 2 & 2 \end{bmatrix}$$, then $$\mathcal{N}(A) = \left\{\begin{bmatrix} -c \\ c\end{bmatrix} \forall c \in \mathbb{R}\right\}$$ and there are infinitely many solutions.

## $$A$$ is skinny ($$m > n$$)
Such a system of linear equations is called _overdetermined_. Prof. Boyd mentions in the lecture notes that such a system cannot be solved for most values of $$y$$. One way to intuitively reason this is to note that the $$n$$ columns of $$A$$ span a low dimensional subspace (of dimension at most $$n$$) in $$\mathbb{R}^m$$. The chance that a random vector $$y \in \mathbb{R}^m$$ lies in the low-dimensional $$\mathcal{C}(A)$$ is slim. Speaking even more loosely, there are way too many constraints (the $$m$$ equations) that the $$n$$ dimensional vector $$x$$ has to satisfy.


In the case where $$y \notin \mathcal{C}(A)$$, no solution to the linear system exists. We can find an approximate solution however. _One possible approximation_ is to find some $$\hat{x} \in \mathbb{R}^n$$ such that the Euclidean distance between $$A\hat{x}$$ and $$y$$ is minimized. Note that $$A\hat{x} \in \mathcal{C}(A)$$, and we want it to be closest to $$y$$ compared to any other point in $$\mathcal{C}(A)$$. Geometrically, such an $$A\hat{x}$$ is the projection of $$y$$ on $$\mathcal{C}(A)$$. This means the error vector $$y - A\hat{x}$$ is orthogonal to $$\mathcal{C}(A)$$ and so $$(y - A\hat{x}) \in \mathcal{N}(A^T)$$. So:

$$
\begin{align}
A^T(y - A\hat{x}) &= 0 \\
\implies \hat{x} &= (A^TA)^{-1}A^Ty
\end{align}
$$

This formula assumes that $$A$$ has full column rank, i.e. $$\text{rank}(A) = n$$. Only then will the matrix $$A^TA$$ be invertible (see point 3 in the Appendix of [my blog post on SVD](https://talwarabhimanyu.github.io/blog/2020/07/10/svd2) for a proof).

## $$A$$ is fat ($$m < n$$)
