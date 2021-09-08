---
layout: post
title: Basic Integrals from Kingma, Welling (2013)
date: 2021-09-08
tags: variational-inference
---
In this post I'll show the calculations behind some integrals presented in the seminal paper "Auto-Encoding Variational Bayes" by Kingma and Welling (2013). The paper presents the results of those integrals, however it does not derive them. I was interested in the derivations and I'm noting them here because someone else may find them useful too. I will not be discussing any other concepts from the paper in this post. We are interested in maximizing the $$\text{ELBO}$$:

$$
\text{ELBO}(x; \theta, \phi) = \underbrace{\mathbb{E}_{q_\phi(z|x)}\left[\log p(x|z;\theta)\right]}_{\text{Expected} \\ \text{Reconstruction} \\ \text{Error}} - D_{KL}\left(q_\phi(z|x) || p_\theta(z)\right)
$$

While we use Monte Carlo to estimate the Expected Reconstruction Error, we can analytically compute the KL Divergence term when the prior $$p_\theta(z)$$ and the variational distribution $$q_\phi(z\|x^{(i)})$$ are Gaussians with a diagonal covariance structure. We want to compute $$D_{KL}\left(q_\phi(z\|x) \|\| p_\theta(z)\right)$$ where $$p_\theta(z) = \mathcal{N}(z; 0, \mathbf{I})$$ and $$q_\phi(z\|x^{(i)}) = \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2})$$. Let $$J$$ be the dimensionality of $$z$$. Then:

$$
\int q_\phi(z|x^{(i)}) \log p_\theta(z)dz = \int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) \log \mathcal{N}(z; 0, \mathbf{I}) dz \\
= \int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) \log \left(\frac{1}{\sqrt{(2\pi)^J}} \exp{\left(-\frac{z^Tz}{2}\right)}\right)dz \\
= - \frac{J}{2}\log {2\pi}\underbrace{\int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) dz}_{1} - \frac{1}{2} \int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) z^Tz dz \\
= - \frac{J}{2}\log {2\pi} - \frac{1}{2} \mathbb{E}_{\mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2})}\left[||z||_2^2\right] \\
= - \frac{J}{2}\log {2\pi} - \frac{1}{2} \mathbb{E}_{\mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2})}\left[\sum_jz_j^2\right] \\
= - \frac{J}{2}\log {2\pi} - \frac{1}{2} \sum_j\mathbb{E}\left[z_j^2\right] \\
= - \frac{J}{2}\log {2\pi} - \frac{1}{2} \sum_j\left(\mathbb{Var}\left[z_j\right] + \mathbb{E}\left[z_j\right]^2\right) \\
= - \frac{J}{2}\log {2\pi} - \frac{1}{2} \sum_j\left(\sigma_j^{(i)2} + \mu_j^{(i)2}\right)
$$

And:

$$
\int q_\phi(z|x^{(i)}) \log q_\phi(z|x^{(i)}) dz = \int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) \log \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) dz \\
= \int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) \log \left(\frac{1}{\sqrt{(2\pi)^J\prod_j\sigma_j^2}} \exp{\left(-\frac{(z-\mu)^T(\sigma^2\mathbf{I})^{-1}(z-\mu)}{2}\right)}\right)dz \\
= \left(- \frac{J}{2}\log {2\pi} -\frac{1}{2}\sum_j\log \sigma_j^{(i)2}\right) \underbrace{\int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2})dz}_{1} - \frac{1}{2} \int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2}) \sum_j \frac{(z_j-\mu_j^{(i)})^2}{\sigma_j^{(i)2}}dz \\
= - \frac{J}{2}\log {2\pi} -\frac{1}{2}\sum_j\log \sigma_j^{(i)2} - \frac{1}{2} \sum_j \frac{1}{\sigma_j^{(i)2}} \underbrace{\int \mathcal{N}(z; \mu^{(i)}, \sigma^{(i)2})  (z_j-\mu_j^{(i)})^2dz}_{\sigma_j^{(i)2}} \\
= - \frac{J}{2}\log {2\pi} -\frac{1}{2}\sum_j\left(1 + \log \sigma_j^{(i)2}\right)
$$

Finally we put this together to compute the KL Divergence term:
$$
-D_{KL}\left(q_\phi(z|x^{(i)}) || p_\theta(z)\right) = \int q_\phi(z|x)\left(\log p_\theta(z) - \log q_\phi(z|x)\right)dz \\
= \frac{1}{2}\sum_{j=1}^J\left(1 + \log \sigma_j^{(i)2} - \sigma_j^{(i)2} - \mu_j^{(i)2}\right)
$$

