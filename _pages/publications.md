---
layout: archive
title: "Current Research (by Area)"
permalink: /research/
author_profile: true
---
***

## 1. Stochastic Algorithm for Nonsmooth Nonconvex-Nonconcave Minmax Problem

**Background:** The nonconvex-nonconcave minmax problem is an important issue arising in operations management and machine learning theory. For example, many operational objectives are formulated in a minmax setting $\min_{x}\max_{y}f(x,y)$, where $x$ and $y$ are real-valued vectors and $f$ is neither (necessarily) convex in $x$ nor (necessarily) concave in $y$. Without solid assumptions on $f$, this problem might be intractable. However, it might be even trickier with complicated constraints for $(x,y)$. There are also a few classical applications for nonconvex-nonconcave minmax problem in machine learning theory, such as the training of generative adversarial networks (GANs) or robust neural-nets. The typical Wasserstein GAN (WGAN) objective is defined as the minimization of the Wasserstein distance between the discriminator $D_y$ of the true distribution $b$ and that of the one learnt by the generator $G_x$ as follows,

$$ \min\limits_x \max\limits_{y} \mathbb{E}_{\beta\sim b} [D_y(\beta)] - \mathbb{E}_{\alpha\sim a}[D_y(G_x(\alpha))],$$

where $G_x$ might not be convex in $x$ for all $y$ and/or $D_y$ not be concave in $y$ for all $x$ in this saddle point problem. There may also be constraints on $x$ and $y$, which results in a complicated constrained nonconvex-nonconcave minmax problem. 


**Research Direction:** To identify the solutions of this type problem, we have <span style="color:DarkGoldenRod">3 research directions</span>. 
1. VI/MI: The minmax optimization problem can be cast as a special case of variational inequality problems (VIPs) or monotone inclusion problems (MIPs). Therefore, by identifying stochastic algorithms that achieve the optimal convergence rate in generalized MIPs or VIPs, we are able to attain the saddle points or stationary points under certain convergence rate for a family of structured nonconvex-nonconcave minmax optimization problems.

2. Weak Assumption: Another line of our work is to ensure the convergence under the weaker assumptions of convexity-concavity, such as the two-sided Polyak-Łojasiewicz condition or $\rho$-weakly convex-concave assumption. 

3. <span style="color:DarkGoldenRod">Normal Map (Main Direction)</span>: Especially for nonsmooth case, we introduce normal mapping to offer a normal map-based method to resolve the biasedness of some stochastic proximal methods like prox-SGDA and to measure the convergence of these methods by converting the expectation of the natural residual into a normal mapped one.

**Normal Map-based Method:** Given proximity is a <span style="color:DarkGoldenRod">nonlinear</span> operator, the unbiasedness of the randome gradient in stochastic proximal methods would be lost. Take the proximal Gradient Descent Ascent (prox-SGDA) as an example,

$$\mathbb{E}_{k}[x^{k+1}]=\mathbb{E}_{k}[\textrm{prox}_{\alpha_{k}\varphi}(x^{k}-\alpha_{k}g^{k})]<span style="color:DarkGoldenRod">\neq \textrm{prox}_{\alpha_{k}\varphi}(x^{k}-\alpha_{k}\nabla f(x^{k})) </span>$$





finding algorithms that achieve certain (optimal) convergence rate in generalized inclusion or VI problem allows us to obtain the same convergence rate for a specific family of structured nonconvex-nonconcave minmax optimization problem.

As a special case of inclusion/vi result, the algorithm enjoys the same convergence rate for solving a non-trivial class of nonconvex-nonconcave minmax optimization problem.

The constrained convex-concave minmax optimization is a special case of constrained single-valued monotone inclusion problems.


To find out the solutions for this type structured problem, one way is to cast the problem as a variational inequality problems (VIPs) or monotone inclusion problems (MIPs). Given a fewer weaker assumptions like weak Minty Variational Inequalities (MVI), generalized monotone inclusion (GMI) do not imply $f$ is convex-concave, we can obtain the same convergence for solving the minmax problem as solving the VIPs or MIPs. Another line of thinking is to introduce the two-sided Polyak-Łojasiewicz condition, which ensure a linear convergence with gradient descent ascent (GDA) algorithm while not require the convex-concave assumption for $f$. Moreover, introducing normal mapping can convert the gap function from the natural residual to a normal map based one. For convenience, the corresponding constraints can be rewritten (w.r.t.) as a proximal problem,

$$\min\limits_x \max\limits_{y} g(x)+\mathbb{E}_{\delta\sim \Delta}[x,y,\delta]-h(y)$$


we can introduce weak Minty Variational Inequalities (MVI) or two-sided Polyak-Łojasiewicz condition


1.background：nonconvex-nonconcave minmax问题是运筹学和机器学习中的重要问题，比如说工业生产中经常会遇到带有约束条件的、且符合f(x,y)对x非凸对y非凹的复杂目标函数，从而难以求解。（并且考虑到数据量大和目标函数呈现求和的形式（可以写一下表达式），往往需要考虑使用随机优化算法进行求解）又如，在机器学习领域中，GAN的训练也是一个经典的随机minmax问题（写一下GAN的表达式），而当generator对x非凸，discriminator对y非凹时，就呈现了一个复杂的Stochastic Nonconvex-Nonconcave Minmax Problem。然而找到该类问题的saddle point是一件非常困难的事，我们可以考虑引入变分不等式、PL condition，或直接把此类有约束的问题写成proximal problem的形式，引入normal map来改写natural residual迭代求解。

2.Direction 1：Normal Map for natural residual （Main direction！）

3.Direction 2：weak Minty Variational Inequality

4.Direction 3：two-sided PL condition


My general research agenda lies in operation research, optimization, and machine learning theory. Regarding operation research, I focus primarily on important issues in supply chain and operation management, including inventory management, outsourcing and procurement, supply chain finance, etc. In addition, my research applies optimization, machine learning, and data-driven decision making to address information and operational problems in business and society. Theoretically, I am interested in making mathematical analysis (optimality conditions, dual gap, error bounds, etc.) for optimization algorithms, and accordingly designing new algorithms with faster convergence and less iteration complexity, escpecially for nonconvex and nonsmooth problem. My study also focuses on applying optimization theory to address some primary issues arising in machine learning algorithms, such as training GANs or robust neural-nets by solving the minmax problem with additional nonsmooth regularizers. 