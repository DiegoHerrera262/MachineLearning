# Advanced Machine Learning Coursera

This repository contains my learning process on Machine Learning. This is a complete specialization on Machine Learning from HSE University. It consists on the following courses:

1. Introduction to Deep Learning.
1. How to Win a Data Science Competition.
1. Bayesian Methods for Machine Learning.
1. Practical Reinforcement Learning.
1. Deep Learning in Computer Vision.
1. Natural Language Processing.
1. Addressing Large Hadron Collider Challenges by Machine Learning.

To see the details of my learning experience, check out my blog at [this link](https://diegoherrera262.github.io/).

## Linear models, regularization and stochastic methods for optimization (week 1)

In this introduction, a problem of object counting on an image is presented. The approach to this problem, from a machine learning perspective, is to take a large dataset of images with their corresponding object count. From this dataset, usually consisting of pairs input-output, a _function_ that goes from the set of all possible input images, to a set of positive numbers is _inferred_. This is commonly known as supervised learning

### Supervised learning: an introduction

Now it is time for some terminology. Any possible input to a given machine learning problem is called _example_, and is denoted by $x_i$. An example that is used to train a particular learning model is called _training example_. Usually, an example contains several _features_.

> An **example** is a vector of $d$ **features**, and is denoted as follows:
> $$x_i = (x_{i1}, x_{i2}, \ldots, x_{id})$$

In supervised learning, a _label_ is associated to a given example. This is sometimes called a _target_, and may represent the expected or true answer of a machine learning problem instance.

> In supervised learning, a **target** is associated to any example unambiguously. The target associated to $x_i$ is denoted by $y_i$.

Now, in problems of supervised learning, inference is based on a _training_ set, from which a decision rule is proposed.

> A **training set** is a set of pairs example-target, which are taken as a starting point for inferring the target of new or unseen examples. It is denoted as:
> $$X = \{(x_i, y_i) \text{ for } i = 1,\ldots,n\}$$

The goal of machine learning algorithms is to come up with the best guess for the function that relates examples and targets, given a training set.

> The guess that a machine learning algorithm produces for matching examples and targets is called a **hypothesis**, which is denoted by:
> $$a(x; X)$$

Which is explicit on the fact that the accuracy of the hypothesis depends not only on the algorithm, but also on the training dataset.

#### Regression and classification

Usually, there are two main kinds of instances of a supervised learning problem, The classification of these rely upon the type of target that the problem instance requires for mathematical modelling. If the target corresponds to a _real valued continuous_ variable, the problem is said to be a _regression problem_. Oh the other hand, if the target corresponds to a _discrete valued integer_ variable, the problem is said to be a _classification_ problem.

When dealing with regression problems, most instances can be very well described by an example-target relationship of the form

$$y = \vec{w} \cdot \vec{x} + b$$

The vector $\vec{w}$ is called **weight** vector, and the quantity $b$ is called the **bias** of the model. For a given training set, the predictions of the hypothesis are described by

$$a(X) = X \cdot \vec{w}$$

Where $X$ is the matrix of all $d$ examples, with $n$ features

$$
X =
\begin{bmatrix}
    x_{11}  & \dots & x_{1n} \\
    \vdots  & \ddots & \vdots \\
    x_{1d}  & \dots & x_{dn} \\
\end{bmatrix}
$$

This produces a vector of _predicted targets_, $\vec{y}'$, that is expected to be the best guess of the actual target vector $\vec{y}$. A common way to assess the quality of the hypothesis is by the **mean squared error**:

$$L(\vec{w}) = \frac{1}{d} |\vec{y}' - \vec{y}|^2 = \frac{1}{d} |X \cdot \vec{x} - \vec{y}|^2$$

**Note:** For now, it is assumed that the bias is cero, since it is possible to find an example for which $a(\vec{x}) = 0$ by a translation.

The less the mean squared error of a hypothesis, the better the predictions. That is the general rule. The natural conclusion is that the set of weights that produce the minimum of mean squared error produces the best hypothesis. In fact, by direct calculation, it is easy to see that the best set of parameters is

$$\vec{w}_{\text{best}} = (X^T X)^{-1}X^T \cdot \vec{y}$$

> However, **computing the inverse of a matrix is expensive**. It is no easy task, and thus, for examples with a large amount of features, impractical. This will be addressed later.

When dealing with binary classification problems, a linear model, such as the one presented before, can be adapted by considering _the sign_ of the dot product. If the dot product is positive, the example is classified in one class (i. e. positive class), otherwise, it is classified as a member of the other class (i. e. negative class). This time, the hypothesis is defined as follows

$$a(\vec{x}) = \text{sign}(\vec{w} \cdot \vec{x})$$

> The geometrical interpretation is that we are finding the best hyperplane that separates the training set into the two target classes. On one side, there is the **positive class**, and on the other, the **negative class**.

Multi-class problems are far more involved. Such instances require that the model predicts the correct target among a set of more than two classes. The common approach is to separate multi-class problems into a sequence of binary classification problem instances.

For example, suppose that we would like to classify an article as a sports, social or politics one. We could proceed by deciding whether the article is about sports or not. Our model would produce an score for such a question. Then, we could ask the question of whether the article is about social topics, or not, and obtain another score. LIkewise with the last category.

> By partitioning a multi-class problem instance into multiple binary classification instances, we decide the target of a given example by the highest score obtained by each of the linear models defined to solve each binary classification.

**Note:** On each subproblem instance, the score is the dot product. The larger the score, the more decisive is the target predicted by the hypothesis.

Mathematically, the hypothesis of a multi-class problem, with $K$ classes, solved by using linear models, is:

$$a(X) = \text{arg } \text{max}_{k = 1, \ldots K} (\vec{w}_{k} \cdot \vec{x})$$

Now, how should we go on to learn a particular set of coefficients for a classification problem? Well, we should define a proper cost function that needs to be minimized. The simplest is the so called **classification accuracy**

$$
\frac{1}{d} \sum [a(x_i) = y_i]
$$

Where the brackets represent the Iverson brackets defined as follows:

$$
[P] =
\begin{cases}
    1 & \text{ if } P \text{ is true} \\
    0 & \text{ if } P \text{ is false}
\end{cases}
$$

So that the classification accuracy is simply the ratio of correctly classified examples to the total number of examples. Although straightforward, a problem of this model is that the loss function is **non differentiable**, which prevents optimization using gradient techniques. Another problem is that the metric does not take into account the **confidence of the model**. The more confident, the better for scaling purposes.

One possibility is to use mean squared error as a cost function. The problem is that this model prevents high confidence (i.e. high absolute value scores). This is not desirable. Then, the best option is to **normalize** a particular set of predicted scores for each of the $K$ classes, so that they can be interpreted as a _probability vector_. The easiest way is using a **softmax transform**

$$
\vec{z} = (\vec{w}_1 \cdot \vec{x}, \ldots, \vec{w}_K \cdot \vec{x}) \rightarrow
\Bigg(\frac{\mathrm{e}^{z_i}}{\sum \mathrm{e}^{z_i}}, \ldots, \frac{\mathrm{e}^{z_K}}{\sum \mathrm{e}^{z_i}}\Bigg)
$$

This is part of the so called **logistic regression** which is a technique in which overconfidence is not penalized. The loss function would simple consist of a summation of the logarithm of the ratio of the expected probability that each example is classified on the correct class (i.e. one) to the predicted probability by a linear classifier. In mathematical therms, the cost is simply:

$$
L(\vec{w}_i, b) = - \sum_i -\text{log}\bigg(\frac{\mathrm{e}^{\vec{w}_{y_i} \cdot \vec{x_i}}}{\sum_k \mathrm{e}^{\vec{w}_{k} \cdot \vec{x}_i}}\bigg)
$$

And remembering that each term on the sum represents the ratio between the expected probability that the correct class is identified, to the actual model-predicted class.

> The loss function proposed in a logistic regression is not easily differentiable by analytic methods. So, most of the algorithms rely on numerical methods for optimization.

### Gradient descent

The technique of gradient descent is pretty straightforward from vector calculus. The idea is to take steps in the direction opposite to the gradient, starting from some random point, in order to find the minima of a loss function. The algorithm can be stated roughly as follows

> 1. Choose a clever starting point $\vec{w}_0$.
> 2. Choose a particular _learning rate_, $\eta_i$ such that, at each point in the descent
>    $$\vec{w}_i = \vec{w}_{i-1} - \eta_i \nabla L(\vec{w}_{i-1})$$
> 3. Continue the descent until a stopping condition is reached. A possible condition can be
>    $$|\vec{w}_i - \vec{w}_{i-1}| < \epsilon$$

Now, there are a lot of heuristics concerning gradient descent that need to be addressed. For instance, how is the gradient to be computed? If analytical methods result in complicated expressions, that is not the way to go. And how is the starting point to be chosen? If it is such that the algorithm converges to local minima, then it should be changed, or at least a better option be considered. Furthermore, is the learning rate supposed to be of constant value at each iteration step? And which value is optimal for a particular cost function and starting point? And equally important, which stopping criterion should be used? Will it produce a minimum in a reasonable amount of time?

### Overfitting in Machine Learning

Consider a machine learning model that has high accuracy _inside a particular training set_. This could lead to the misleading conclusion that the model will correctly label any example, even outside the training set.

> In general, high accuracy in the training set doesn't imply that a model **generalizes** correctly. This may be an instance of an **overfitting** condition.

Overfitting is akin to the smart kid in class that always repeats the answers of the book, but never really understands why is going on. The guy simply splits the information received, but nothing more.

Consider as an example the problem of fitting a non-linear function of one variable. If regular at $x=0$, then it can be expanded in Taylor series:

$$f(x) = f_0 + a_1 x + a_2 x^2 + \cdots$$

In this case, linear regression may not enough for fitting any discrete of points of a set of pairs $(x_1, f(x_i))$. A better approach would be to use a _polynomial regression_. This would yield a hypothesis

$$\bar{f}(x) = \bar{f}_0 + \bar{a}_1 x + \bar{a}_2 x^2+ \cdots + \bar{a}_n x^n$$

For some integer $n$. Now, this may seem reasonable. But consider $f(x)$ a polynomial. Well, clearly, if $n$ is greater than the actual degree of $f$, the model would have more parameters than needed to fit de training set data. It is possible to find a hypothesis function that fits quite well the target function on the training set. Yet, it would fail miserably on examples outside. Plus, this is not only the case for polynomial functions, in neighborhoods of some point, a non-linear function can be very well overfitted by a large degree polynomial.

How can we assess if a model is overfitted? The trick is the definition of _holdout sets_. These would be set apart from the total number of input examples for training. We would define the rest as _the training set_ for the model. We can use the training set for finding the optimal parameters, and the holdout set, as a sort of simulation of real life examples. We can then use a _quality metric_ (cross-entropy, accuracy, mean squared error, etc.) to assess the quality of the model.

> If the quality measure inside the **holdout set** does not fall dramatically to that of the actual training set, the model is thought to be reliable.

Now, it is not trivial how to partition the total set of input examples so as to accurately evaluate the confidence of a model, and to accurately find optimal parameter values for the model. In practice, 70%-80% is put in the actual training set, whereas the rest is put inside the holdout set.

However, for small sets of input examples, it is desirable to evaluate the quality of a model when its examples have both the roles of training and holdout. We can solve this issue by the technique of _cross validation_. In that case, whe create a partition of $K$ subsets called _folds_, and train the model with each subset as holdout and the rest as training. Then we estimate the quality of the model with the average quality metric over the folds. Cross validation is often impractical in deep learning.

> The key is to use a large enough dataset such that the size of the holdout set is representative.

### Model regularization

How can we reduce the complexity of a particular machine learning model, and how to save i from overfitting? Well, it is possible to start by noting that if model parameters are to large in magnitude, the model is probably overfitted. Hence, we can penalize large parameter values in the cost function.

> We can extend the loss function of a modal with a **regularization function** $R(\vec{w}_i)$, weighted by a regularization strength $\lambda$ such tat the regularized cost function has the shape
> $$\bar{L}(\vec{w}_i) = L(\vec{w}_i) + \lambda R(\vec{w}_i)$$
> Regularization can be viewed as a **constrained optimization problem**, where the constraint is that the regularization function remains bounded by some constant.

The simplest regularized cost function is called _L2 penalty_, and has the shape

$$\bar{L}(\vec{w}_i) = L(\vec{w}_i) + \lambda ||\vec{w}_i||^2$$

This penalty is straightforward to implement, differentiable, and penalizes high values of the parameter models. ANother common regularized cost or loss function is the _L1 penalty_, and is simply

$$\bar{L}(\vec{w}_i) = L(\vec{w}_i) + \lambda |\vec{w}_i|^2$$

Where we define

$$|\vec{a}| = \sum_i |a_i|$$

Although this cost function is not differentiable on all points of space, it has a nice property: it tends to drive some parameters of the model exactly to zero. This can be useful to find which parameters of a model are most relevant in some machine learning problem instance. However, advanced optimization methods are needed to find the model parameters tha fit best.

Now, there are other techniques for preventing overfitting. One is using a larger dataset, or stop gradient descent early. Also, some data preprocessing can hep: dimensionality reduction, principal component analysis, etc.

### Stochastic methods for optimization

If a particular training set is large (millions of samples) or we wan a real time learning application, it is impractical tom compute te gradient of cost function directly. However, it may be noticed that most (regularized) cost functions can be separated into sums of lost functions that involve individual examples. This is the case, for example, of mean squared error and logistic models.

> In stochastic optimization, we assume that the gradient of the entire cost function is in the direction of the gradient of any of its single-example constituents, chosen randomly. We then walk in that direction to optimize.

Albeit efficient, stochastic gradient descent depends critically on the _learning rate_. If it is too large, the method cannot converge. If it is too small, the number of iterations required for convergence will be huge. To overcome this, we can chose a _mini batch_ of $m$ random examples, and compute the gradient of the component of the cost function associated to the set. Both stochastic and mini batch gradient descent produce noisy convergence to a minimum, but mini batch i usually less noisy.

A problem of stochastic methods is that for some set of functions, the stochastic gradient does not follow the optimal path for converging to minimum, which implies that the number of iterations for obtaining an optimal set of parameters is large. Now, let's discuss some advanced optimization techniques that extend gradient optimization.

#### Advanced optimization techniques

The goal of this methods is to modify the update rule in a gradient optimization algorithm in order to make it converge faster. It can take the gradient as computed by an stochastic method, mini-batch method, or even exact gradient. Let's denote the gradient by $\vec{g}_t$

The first technique discussed here is _momentum optimization_. The update rule for this algorithm updates the model parameters not with the stochastic gradient itself, but with an sor of weighted average over the iteration steps in the past:

$$
\vec{h}_i = \alpha \vec{h}_{i-1} + \eta_i \vec{g}_i  \\
\vec{w}_i = \vec{w}_{i-1} - \vec{h}_i
$$

The basic intuition behind this technique relies on a pretty rough understanding on how stochastic gradient descent would work. On a difficult loss function surface, stochastic gradient would bounce back and forth (like velocity vector of a falling object in a mountain), till reaching a minimum. Now this bouncing is undesirable, since it increases the amount of iterations required to converge to a minimum. What is even worse, the bounce can be so large that the algorithm fails to converge. A basic way to prevent this former effect is by reducing the learning rate, at the cost of extending the algorithm convergence. Momentum, on the other hand, averages the gradient at different iterations. The effect is that strong oscillations tend to cancel out, whereas the averaged vector points strongly to the direction of real convergence.

Another related technique is called _Nesterov momentum optimization_. The idea is to update $\vec{h}$ with the gradient computed not at $\vec{w}$, but at an intermediate point. My basic intuition on his algorithm is tat it is kind of like when you are walking on a rocky field, and you some sort of stick to see where the point you expect to move leads you. Then, you can decide whether you should actually move to the point, or simply take an equivalent shortcut. The update rules are these:

$$
\vec{h}_i = \alpha \vec{h}_{i-1} + \eta_i \vec{g}(\vec{w}_{i-1}-\alpha \vec{h}_{i-1})  \\
\vec{w}_i = \vec{w}_{i-1} - \vec{h}_i
$$

Still, both _momentum_ and _Nesterov momentum_ require careful consideration of the value of the learning rate and the parameter $\alpha$. There are other optimization methods that would help speeding up optimization with adaptive learning rate definition.

One of such methods is _AdaGrad_. In this algorithm, we update each of the parameters sort of independently. This is very good for sparse data. Instead of updating a sort of velocity vector, like momentum, we update the learning rate according to the gradient in the direction of a particular parameter. The update rules for this algorithm are

$$
G_j^{t} = G_j^{t-1} + (g_j^{t})^2 \\
w_j^{t} = w_j^{t-1} - \frac{\eta_t}{\sqrt{G_j^{t} + \epsilon}} g_j^{t}
$$

Another method is _RMSprops_ which extends AdaGrad. The idea is to adapt the learning rate with an _exponentially weighted_ sum of the square of the gradients. The update rule is:

$$
G_j^{t} = \alpha G_j^{t-1} + (1-\alpha)(g_j^{t})^2 \\
w_j^{t} = w_j^{t-1} - \frac{\eta_t}{\sqrt{G_j^{t} + \epsilon}} g_j^{t}
$$

AdaGrad has a problem with large gradients, which can lead to a slow convergence. RMSprops can overcome this problem. There is an algorithm that combines both RMSprops and momentum in a clever way. It is called _Adam_, and has a momentum-like rule, as well as a RMSpros-like rule (both with a bias correction). The update rules look like this:

$$
m_j^{t} = \frac{\beta_1 m_j^{t-1} + (1-\beta_1)g_j^{t}}{1-\beta_1^t} \\
v_j^{t} = \frac{\beta_2 v_j^{t-1} + (1-\beta_2)(g_j^{t})^2}{1-\beta_2^t} \\
w_j^{t} = w_j^{t-1} - \frac{\eta_t}{\sqrt{v_j^{t} + \epsilon}} m_j^{t}
$$

This method is quite used in machine learning applications. It is very respected in the field.
