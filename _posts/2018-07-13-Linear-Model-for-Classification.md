---
title: Linear Model for Classification
date: 2018-07-13
categories: [Notes, Machine Learning]
tags: [ML, 线性模型, 分类, 算法]
math: true
---

## Summary of Error Function

All three algorithm we mentioned above have the same linear scoring function $s=\boldsymbol w^{\text T}\boldsymbol x$, now we discuss about the similarity of these algorithm in binary classification $\mathcal Y\in \lbrace-1,+1\rbrace$.

- Perceptron

  $$
  \begin{split}
  h(\boldsymbol x)&=\text{sign}(s) \\[1em]
  \text{err}_{0/1}(s,y)&=[\![\text{sign}(s) \ne y]\!] \\[1em]
  &=[\![\text{sign}(ys) \ne +1]\!]
  \end{split}
  $$

- Linear rergreesion

  $$
  \begin{split}
  h(\boldsymbol x)&=s \\[1em]
  \text{err}_{\text{SQR}}(s,y)&=(s-y)^2 \\[1em]
  &=(ys-1)^2
  \end{split}
  $$

- Logistic regression

  $$
  \begin{split}
  h(\boldsymbol x)&=\sigma(s) \\[1em]
  \text{err}_{\text{CE}}(s,y)&=\ln(1+\exp(-ys)) \\[1em]
  \text{scaled:} \ \ \text{err}_{\text{SCE}}(s,y)&=\log_2(1+\exp(-ys))
  \end{split}
  $$

Draw these error function on an image:

<img src='https://raw.githubusercontent.com/weirme/picme/main/3.png' width='50%'>

as shown, $\rm err_{SCE}$ is an upper bound of $\rm err_{0/1}$, therefore if we can limit $\rm err_{SCE}$ to a small value, then $\rm err_{0/1}$ should also be small. And VC bound theory gives a more powerful proof of it. Thus we can safely use linear models for classification.

Finally, we compare the pros and cons of these algorithm in classification:

+ Perceptron
  + Pros: $\text{err}_{0/1}$ can be reduced to the lowest value if dataset is linear separable.
  + Cons: deteriorated to a NP-hard problem if dataset is non-linear separable.
+ Linear regression
  + Pros: easy to do optimize
  + Cons: it cannot ensure $\text{err}_{0/1}$ is small enough when $ys$ is very small or very large.
+ Logistic regression
  + Pros: comparably easy to optimize
  + Cons: it cannot ensure $\text{err}_{0/1}$ is small enough when $ys$ is very small.

## Stochastic Gradient Descent

We have known both PLA and logistic regression realize optimization through iteration, that is

**For** $t=0,1,...$  **do**

$$
\boldsymbol w_{t+1} \gets \boldsymbol w_t+\eta\boldsymbol v \tag{6.1}
$$

When stop, return $\boldsymbol w$ as $g$.

Difference is that each iteration of PLA correct only one point, while logistic regression calculates the contribution to gradient of every point and take the average, so each iteration has a complexity of $O(n)$.

To reduce the complexity, we select a point randomly and calculates its contribution to gradient $\nabla_{\boldsymbol w}\text{err}(\boldsymbol w, \boldsymbol x_n, y_n)$, called **Stochastic Gradient**. And we can regard the real gradient as the expectation of stochastic gradient. From another side, we also regard the stochastic gradient as sum of real gradient and a zero-mean noise. In this way, we replace real gradient with stochastic gradient, called **Stochastic Gradient Descent**. After enough iteration, we can also get a acceptable result. Each iterative formula as follows:

$$
\boldsymbol w_{t+1} \gets \boldsymbol w_t+\eta \cdot \boldsymbol \sigma(-y_n\boldsymbol w^{\text T}\boldsymbol x)(y_n\boldsymbol x_n) \tag{6.2}
$$

Look at the iterative formula of PLA again:

$$
\boldsymbol w_{t+1} \gets \boldsymbol w_t+1\cdot [\![y_n \ne \text{sign}(\boldsymbol w^\text{T}\boldsymbol x)]\!] y_n\boldsymbol x_n \tag{6.3}
$$

where $y_n \ne \text{sign}(\boldsymbol w^\text{T}\boldsymbol x)$ means PLA update $\boldsymbol w$ only when an error point $\boldsymbol x_n$ founded. From the two formula above, SGD logistic regression can be regard as a 'soft' PLA, because in PLA we take whether $y_n \ne \text{sign}(\boldsymbol w^\text{T}\boldsymbol x)$ as aJudging criteria, while in SGD logistic regression, we only consider how close they are. By the way, if we ensure $\boldsymbol w^{\text T}\boldsymbol x$ is big enough, then $\eta=1$ may be a good choice.

When using SGD logistic regression, we usually stop iteration when $t$ is large enough and use $\eta \approx 0.1$ when $\boldsymbol x$ in proper range.

## Muticlass Classification

### One-Versus-All

Assuming that there are four categories in the question, each time we treat one of the categories as positive , while all other categories become negative, it can build a binary classifier. Keeping loop, and finally we will get 4 classifiers as following figure:

<img src='https://raw.githubusercontent.com/weirme/picme/main/4.png' width='80%'>

But when all the four classifiers put a sample into negative class, such as the middle area of the figure above. what should we do?

It's easy if we use logistic regression and set each classifier output a probality of a sample belonging to its own category. We can simply choose the highest probability of each classifier as the final result.

$$
\begin{split}
g(\boldsymbol x) &= \mathop{\arg\min}_{k\in \mathcal Y} \ \sigma(\boldsymbol w^{\text T}_{[k]}\boldsymbol x) \\[1em]
&=\mathop{\arg\min}_{k\in \mathcal Y} \ \boldsymbol w^{\text T}_{[k]}\boldsymbol x \ \ \ \ (\sigma\text{ is monotonic})
\end{split}
\tag{6.4}
$$

We call this method **One-Versus-All (OVA)** decomposition.

### One-Versus-One

It is important to note OVA can't perform very well when $\vert\mathcal Y\vert$ is large. Considering this situation when $\vert\mathcal Y\vert=100$ and number of samples in each category is not much different. If we choose a classifier which put all samples into negative class, then its accuracy rate has reached 99%. Therefore in OVA we might get 100 classifiers each of which judge all samples as negative.

We can use **One-Versus-One (OVO)** decomposition to solve the problem. OVO chooses only two of all categories to bulid a classifier as follows:

<img src='https://raw.githubusercontent.com/weirme/picme/main/5.png' width='80%'>

 after bulid all classifiers (whose number should be $\binom {\vert\mathcal Y\vert}{2}$), we use 'voting method' to get the final result.
