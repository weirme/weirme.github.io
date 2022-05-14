---
title: Adaptive Boosting
date: 2018-08-13
categories: [Notes, Machine Learning]
tags: [ML, Boosting, 算法]
math: true
---

AdaBoost is an algorithm that promotes weak learners to a strong learner. This algorithm first trains a base learner from the initial training set, and then adjusts the sample distribution according to its performance, so that samples where previous learner makes mistake will receive more attention later. Then train the next learner based on the adjusted sample distribution. Repeatedly, and get the final learner by weighting all the learners.

With a parameter $u_n^{(t)}$ represents the degree to which sample $\boldsymbol x_n$ is concerned in $t$-th iteration. Firstly we can assume all $u_n^{(1)} = \frac{1}{N}$. Our goal is to minimize the traning error

$$
E_{\text{in}} = \frac{1}{N}\sum_{n=1}^Nu_n\cdot \text{err}(y_n,h(\boldsymbol x_n)) \tag{10.1}
$$

each iteration to find the optimal $g$ can be done with SVM or logistic regression and so on. If we use $0/1$ error in $(10.1)$, then the optimal $g$ can be written as

$$
g_t= \ \mathop{\arg\min}_{h\in \mathcal H}\sum_{n=1}^Nu_n^{(t)} [\![y_n \ne h(\boldsymbol x)]\!] \tag{10.2}
$$

after finding a $g_t$, we adjust $u_n$ to get $g_{t+1}$. In the process, we make $u_n$ of   samples $\boldsymbol x_n$ where $g_t$ makes mistake large as possible. Record the error rate of $g_t$ as $\epsilon_t$, we construct degree factor

$$
\text{deg}_t=\sqrt{\frac{1-\epsilon_t}{\epsilon_t}} \tag{10.3}
$$

for each correct $\boldsymbol x_n$, we set $u_n^{t+1}  \gets \frac{1}{\text{deg}_t} u_n^t$, while for each error $\boldsymbol x_n$, we set $u_n^{t+1} \gets \text{deg}_t \cdot u_n^t$. Next we look at the degree factor, when $0<\epsilon_t<\frac{1}{2}$, which means $g$ works not bad in training set, and in this condition we have $\text{deg}_t>1$. Through updating $u_n$, we scale up error points and scale down correct points. When $\epsilon_t=\frac{1}{2}$, which means $g$ is so bad with a correct rate same as coin flipping, then we find $\text{deg}_n=1$, so it does nothing when update $u_n$. Finally when $\frac{1}{2}<\epsilon_t<1$, it looks so terrible, but if we 'reserve' this learner, that is consider samples $g(\boldsymbol x_n)=1$ as negative while samples $g(\boldsymbol x_n)=-1$ as positive,  then its correct error is $1-\epsilon_t$, which seems not too bad. In this condition we have $0<\text{deg}_t<1$, through   updating $u_n$, we scale up correct points (actually wrong points) and scale down error points (actually correct points).

In addition, our final hypotheis

$$
G(\boldsymbol x) = \text{sign}\left(\sum_{t=1}^T\alpha_tg_t(\boldsymbol x)\right) \tag{10.4}
$$

where $\alpha_t$ can be also got during each iteration, we set

$$
\alpha_t = \ln\sqrt{\frac{1-\epsilon_t}{\epsilon_t}} \tag{10.5}
$$

reason for choosing this $\alpha_t$ is similar to the degree factor, which used to describe the contribution of each learner.

In theory, we have a conclusion that if each learner is better than coin flipping, only after $T=O(\log N)$ iteration we can reduce $E_{\text{in}}(G)$ approximately equal to $0$.

In action, a popular choice of each $g$ is 'decision stump', that is

$$
h_{s,i,\theta}(\boldsymbol x)=s\cdot \text{sign}(x_i-\theta) \tag{10.6}
$$

where $s,i,\theta$ represent direction, feature and threshold respectively. This algorithm makes a positive and negative rays on $i$-th feature dimension each time. Perhaps this algorithm is too weak to work by itself, but after AdaBoost, it can be really strong.