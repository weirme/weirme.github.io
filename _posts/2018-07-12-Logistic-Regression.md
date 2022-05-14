---
title: Logistic Regression
date: 2018-07-12
categories: [Notes, Machine Learning]
tags: [ML, 逻辑回归, 算法]
math: true
---

In some cases, we want to learn the probability of something happening.   In order to meet this goal, we can use a sigmoid function $\sigma$ which map the result of linear regression to a range between $0-1$, One of the most commonly used $\sigma$ is **Logistic Function**:

$$
\sigma(x)=\frac{1}{1+\text e^{-x}} \tag{5.1}
$$

then we have hypotheses of the form:

$$
h(\boldsymbol x) = \frac{1}{1+\text e^{-\boldsymbol w^{\text T}\boldsymbol x}} \tag{5.2}
$$

Next we try to measure the error function of the hypotheses. Consider the target function representing the probability of $y=+1$, that is $f(\boldsymbol x)=P(+1\vert\boldsymbol x)$, we have

$$
P(y|\boldsymbol x)=
\begin{cases}
f(\boldsymbol x) & y=+1 \\ \\
1-f(\boldsymbol x) & y=-1
\end{cases}
\tag{5.3}
$$

Consider a dataset $\mathcal D=\lbrace(\boldsymbol x_1,+1),(\boldsymbol x_2,-1),...,(\boldsymbol x_N,-1)\rbrace$, we can calculate the probality that $f$ generates $\mathcal D$

$$
\begin{split}
P(f) &= P(\boldsymbol x_1)P(+1|\boldsymbol x_1) \times P(\boldsymbol x_2)P(-1|\boldsymbol x_2) \times \cdots \times P(\boldsymbol x_N)P(-1|\boldsymbol x_N) \\[1em]
&=P(\boldsymbol x_1)f(\boldsymbol x_1) \times P(\boldsymbol x_2)(1-f(\boldsymbol x_2)) \times \cdots \times P(\boldsymbol x_N)(1-f(\boldsymbol x_N))
\end{split}
\tag{5.4}
$$

Similarly, for a single hypothesis $h$, we have the probability that $h$ generates $\mathcal D$ called **Likelihood**

$$
\text{likelihood}(h)=P(\boldsymbol x_1)h(\boldsymbol x_1) \times P(\boldsymbol x_2)(1-h(\boldsymbol x_2)) \times \cdots \times P(\boldsymbol x_N)(1-h(\boldsymbol x_N)) \tag{5.5}
$$

Then if $h \approx f$, it should have $\text{likelihood}(h) \approx P(f)$. Actually we know the $f$ as our target function, so the probability using $f$ generates $\mathcal D$ is usually large. So our goal is to maximum the likelihood of $h$, and the final hypothesis $g$ can be founded

$$
g=\mathop{\arg\max}_h \  \text{likelihood}(h) \tag{5.6}
$$

In addition, for logistic function we have a property as follows:

$$
1-\sigma(x)=\sigma(-x) \tag{5.7}
$$

according to this property, we can rewrite $(5.5)$

$$
\begin{split}
\text{likelihood}(h)&=P(\boldsymbol x_1)h(+\boldsymbol x_1) \times P(\boldsymbol x_2)h(-\boldsymbol x_2) \times \cdots \times P(\boldsymbol x_N)h(-\boldsymbol x_N) \\[1em]
&=\prod_{n=1}^NP(\boldsymbol x_n)h(y_n\boldsymbol x_n)
\end{split}
\tag{5.8}
$$

another hand, $P(\boldsymbol x_n)$ is already known, so we have

$$
\text{likelihood}(h)\propto \prod_{n=1}^Nh(y_n\boldsymbol x_n) \tag{5.9}
$$

Replace $h$ with $\boldsymbol w$ as argument and replace products with summation to avoid overflow, we can do following modification:

$$
\begin{split}
\max_h\prod_{n=1}^Nh(y_n\boldsymbol x_n) &= \max_{\boldsymbol w} \sum_{n=1}^N\ln\sigma(y_n\boldsymbol w^{\text T}\boldsymbol x_n) \\[1em]
&= \min_{\boldsymbol w} \sum_{n=1}^N-\ln\sigma(y_n\boldsymbol w^{\text T}\boldsymbol x_n) \\[1em]
&= \min_{\boldsymbol w} \sum_{n=1}^N\ln(1+\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n})
\end{split}
\tag{5.10}
$$

Record $\frac{1}{N}\sum_{n=1}^N\ln(1+\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n})$ as $E(\boldsymbol w)$ which represents the mean error function of the whole dataset. Fortunately, it is continuous, differentiable, twice-differentiable and convex. Similar to before, all we need to do is making the gradient $\boldsymbol 0$.

According to the chain-rule, we have partial derivation of $E(\boldsymbol w)$:

$$
\begin{split}
\frac{\partial E(\boldsymbol w)}{\partial \boldsymbol w} &=
\frac{1}{N}\sum_{n=1}^N\frac{\partial \ln(1+\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n})}{\partial (1+\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n})} \ \cdot \ \frac{\partial (1+\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n})}{\partial{-y_n\boldsymbol w^{\text T}\boldsymbol x_n}} \ \cdot \ \frac{\partial{-y_n\boldsymbol w^{\text T}\boldsymbol x_n}}{\partial \boldsymbol w} \\[1em]
&=\frac{1}{N} \sum_{n=1}^N\frac{\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n}}{1+\text e^{-y_n\boldsymbol w^{\text T}\boldsymbol x_n}} \ \cdot \ -y_n\boldsymbol x_n \\[1em]
&=\frac{1}{N} \sum_{n=1}^N\sigma(-y_n\boldsymbol w^{\text T}\boldsymbol x_n)(-y_n\boldsymbol x_n)
\end{split}
\tag{5.11}
$$

Obviously, equation $\nabla_{\boldsymbol w}E=0$ has no closed-form solution. For such problems, we usually solve it using iterative optimization. In each iteration, we update $\boldsymbol w_{t+1}$ with $\boldsymbol w_t+\eta\boldsymbol v$, where $\eta$ is called learning rate and $\boldsymbol v$ is a unit vector representing the direction of each update.

For logistic regression, we use an optimization method called **Gradient Descent** to find the optimal solution.

According Taylor expansion, in a neighbourhood of $\boldsymbol w$ we have

$$
E(\boldsymbol w+\eta\boldsymbol v) \approx E(\boldsymbol w)+\eta\boldsymbol v^{\text T}\nabla E(\boldsymbol w) \tag{5.12}
$$

Based on the above derivation, our optimization goal is $E(\boldsymbol w)$. In each iteration, it can rewrite as follows:

$$
\min E(\boldsymbol w+\eta\boldsymbol v)=\min \ (E(\boldsymbol w)+\eta\boldsymbol v^{\text T}\nabla E(\boldsymbol w)) \tag{5.13}
$$

$\boldsymbol w$ and $\nabla E(\boldsymbol w)$ in current iteration is known, $\eta$ is a given positive number and $\Vert\boldsymbol v\Vert=1$, the only thing we need to determine is the direction of $\boldsymbol v$. So our goal can be simplify as

$$
\min_{\boldsymbol v} \ \boldsymbol v^{\text T}\nabla E(\boldsymbol w) \tag{5.14}
$$

Obviously, only if $\boldsymbol v$ and $\nabla E(\boldsymbol w)$ have the opposite directions can this formula get the minimum value. So we have

$$
\boldsymbol v=-\frac{\nabla E(\boldsymbol w)}{\Vert\nabla E(\boldsymbol w)\Vert} \tag{5.15}
$$

and the iterative formula of $\boldsymbol w$ is $\boldsymbol w_{t+1} \gets \boldsymbol w_t-\eta\frac{\nabla E(\boldsymbol w)}{\Vert\nabla E(\boldsymbol w)\Vert}$.

About the choice of learing rate $\eta$, both too large or too small are not a good choice. $\eta$ is better to be monotonic of $\Vert\nabla E(\boldsymbol w)\Vert$. Set $\eta \gets \frac{\eta}{\Vert\nabla E(\boldsymbol w)\Vert}$, we have a new iterative formula $\boldsymbol w_{t+1} \gets \boldsymbol w_t-\eta \nabla E(\boldsymbol w)$.
