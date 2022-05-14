---
title: Linear Regression
date: 2018-07-12
categories: [Notes, Machine Learning]
tags: [ML, 线性回归, 算法]
math: true
---

Given a dataset $\mathcal D = \lbrace(\boldsymbol {x_i},y_i)\rbrace_{i=1}^N$, where $\boldsymbol x \in \mathbb R^d$. we try to get a weight vector $(w_1,w_2,...,w_d)^\text T$ such that

$$
y_i \approx \sum_{i=1}^d w_ix_i + b \tag{4.1}
$$

Set $\boldsymbol w =(b,w_1,...,w_d)^\text T$, $\boldsymbol x=(1,x_1,...,x_d)^\text T$. Then the linear combination of $\boldsymbol w$ and $\boldsymbol x$ is a hypothesis recorded as  $h(\boldsymbol x)$.

Based on least square method, we have squared error $E(\hat y,y)=(\hat y-y)^2$. And our goal is to minimize it. Next we do some calculations as follows:

$$
\begin{split}
E(\boldsymbol w) &= \frac{1}{N}\sum_{n=1}^N \left(\boldsymbol w^\text T \boldsymbol x_n - y_n\right)^2 \\ \\
&=\frac{1}{N}\left|\left|
\begin{array}{ccc}
\boldsymbol x_1^{\text T} \boldsymbol w - y_1 \\
\boldsymbol x_2^{\text T} \boldsymbol w - y_2  \\
\vdots \\
\boldsymbol x_N^{\text T} \boldsymbol w - y_N  \\
\end{array}
\right|\right|^2 \\ \\
&=\frac{1}{N}||\boldsymbol {Xw-y}||^2
\end{split}
\tag{4.2}
$$

where $\boldsymbol X=(\boldsymbol x_1^\text{T};\boldsymbol x_2^\text{T};...;\boldsymbol x_N^\text{T})$, $\boldsymbol y=(y_1,y_2,...,y_N)^\text T$.

Then our goal is

$$
\min_{\boldsymbol w}\frac{1}{N}||\boldsymbol {Xw-y}||^2 \tag{4.3}
$$

The target function $E(\boldsymbol w)$ is continuous, differentiable and convex. So we have the neccessary condition of optimal $\boldsymbol w$ :

$$
\nabla E(\boldsymbol w)=\boldsymbol 0 \tag{4.4}
$$

After derivation, we have

$$
\frac{2}{N}(\boldsymbol {X}^\text T\boldsymbol {Xw}-\boldsymbol {X}^\text T\boldsymbol y)=\boldsymbol 0 \tag{4.5}
$$

If matrix $\boldsymbol {X}^\text T\boldsymbol {X}$ is invertible, it's easy to get the solution of $(4.5)$:

$$
\boldsymbol w_{\text{lin}}=(\boldsymbol {X}^\text T\boldsymbol {X})^{-1}\boldsymbol {X}^\text T\boldsymbol y \tag{4.6}
$$

Even if $\boldsymbol {X}^\text T\boldsymbol {X}$ is a singular matrix,  No need to worry too much because most of programs computing inverse matrix can deal with it easily.
