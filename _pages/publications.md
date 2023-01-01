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

**Normal Map-based Method:** We study the following nonsmooth nonconvex-nonconcave minmax problem

$$
\min\limits_x \max\limits_{y}\ \  \psi(x,y) = \mathbb{E}_{\delta\sim \Delta}[f(x,y;\delta)]+\varphi(x)-h(y),
$$

where $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is Lipschitz smooth and not (necessarily) convex-concave in $z=(x,y)$, while $\varphi: \mathbb{R}^n \rightarrow \mathbb{\bar{R}}$ and $h: \mathbb{R}^n \rightarrow \mathbb{\bar{R}}$ are convex in $x$ and $y$, respectively. The proximity operator of $\varphi$ at $x$ and $h$ at $y$ can be expressed as, 

$$
\begin{align}
\textrm{prox}_{\alpha_{k}\varphi}(x) &:= \textrm{arg}\min_{p\in \mathbb{R}^n} \{\varphi(p)+ \frac{1}{2\alpha}||x-p||^2 \},\\
\textrm{prox}_{\alpha_{k}h}(y) &:= \textrm{arg}\max_{q\in \mathbb{R}^n} \{-h(q)- \frac{1}{2\alpha}||y-q||^2 \}.
\end{align}
$$



Given the above proximities are <span style="color:DarkGoldenRod">nonlinear</span> operators, the unbiasedness of the random gradient in stochastic proximal methods would be lost. Consider the proximal Stochastic Gradient Descent Ascent (prox-SGDA) as an example,

$$
\begin{align}
\mathbb{E}_{k}[x^{k+1}]&=\mathbb{E}_{k}[\textrm{prox}_{\alpha_{k}\varphi}(x^{k}-\alpha_{k}g_{x}^{k})] \color{DarkGoldenRod}{\neq \textrm{prox}_{\alpha_{k}\varphi}(x^{k}-\alpha_{k}\nabla_{x} f(x^{k},y^{k}))},\\
\mathbb{E}_{k}[y^{k+1}]&=\mathbb{E}_{k}[\textrm{prox}_{\alpha_{k}h}(y^{k}+\alpha_{k}g_{y}^{k})] \color{DarkGoldenRod}{\neq \textrm{prox}_{\alpha_{k}h}(y^{k}+\alpha_{k}\nabla_{y} f(x^{k},y^{k}))},
\end{align}
$$

which cannot recover prox-GDA due to the nonlinear proximity operator. To address the biasedness of the stochastic estimator in prox-SGDA, we thus seperate proximity operators from the update process and introduce the <span style="color:DarkGoldenRod">normal map</span> to capture the gradient and subgradient of $\psi$. Let $x := \textrm{prox}_{\lambda \varphi} (u)$ and $y:=\textrm{prox}\_{\lambda h} (v)$, then the normal maps are given as,

$$
\begin{align}
F_{nor}^{\lambda}(u)&: = \nabla_x f(x,y)+\frac{1}{\lambda}(u-x) \in \nabla_x f(x,y)+ \partial \varphi(x),\\
F_{nor}^{\lambda}(v)&: = \nabla_y f(x,y)-\frac{1}{\lambda}(v-y) \in \nabla_y f(x,y)- \partial h(y).
\end{align}$$

When $F_{nor}^{\lambda}(u) = F_{nor}^{\lambda}(v) = 0$, $z = (x,y)$ is the <span style="color:DarkGoldenRod">stationary point</span> of $\psi$. Based on $F_{nor}^{\lambda}(u)$ and $F_{nor}^{\lambda}(v)$, we then define the normal maps as new descent and ascent directions for $u$ and $v$, with additional proximal updating to $x$ and $y$. Thus, we provide the following <span style="color:DarkGoldenRod">Normal Map-based SGDA Algorithm</span>, which can be seen as a special <span style="color:DarkGoldenRod">subgradient</span> method,

$$
\textrm{Loop:}
\left\{\begin{matrix}
\begin{aligned}
&g^k \approx \nabla f(x^k,y^k) \ \ \ \ \ \ \ \ \   \ \  \ \   \   \ \  \color{DarkGoldenRod}{\textrm{by some sampling scheme}} \\
&u^{k+1} = u^k-\alpha_k (g_x^k + \frac{1}{\lambda}(u^k-x^k)) \color{DarkGoldenRod}{\approx u^k - \alpha_k F_{nor}^{\lambda}(u^k)}\\
&v^{k+1} = v^k+\alpha_k (g_y^k - \frac{1}{\lambda}(v^k-y^k)) \color{DarkGoldenRod}{\approx v^k + \alpha_k F_{nor}^{\lambda}(v^k)}\\
&x^{k+1} = \textrm{prox}_{\lambda \varphi} (u^{k+1})\\
&y^{k+1} = \textrm{prox}_{\lambda h} (v^{k+1})
\end{aligned}
\end{matrix}\right.
\tag{9}
$$

where the updates are <span style="color:DarkGoldenRod">unbiased</span>, i.e., 

$$
\begin{align}
\mathbb{E}_{k}[u^{k+1}] &= u^k-\alpha_k (\nabla_{x}f(x^k,y^k) + \frac{1}{\lambda}(u^k-x^k)), \tag{10}\\
\mathbb{E}_{k}[v^{k+1}] &= v^k+\alpha_k (\nabla_{y}f(x^k,y^k) - \frac{1}{\lambda}(v^k-y^k)). \tag{11}
\end{align}
$$

Given the proposed Normal Map-based SGDA Algorithm, a <span style="color:DarkGoldenRod">convergent</span> and <span style="color:DarkGoldenRod">optimistic</span> version can also be formulated by replacing $g^k$ in $(9)$ with $g^{k+1}$ approximated by $(1+\frac{\alpha}{\beta})g^{k}+\frac{\alpha}{\beta}g^{k-1}$ (or a Normal Map-based <span style="color:DarkGoldenRod">Extragradient</span> Algorithm by replacing $g^k$ with $g^{k+\frac{1}{2}}$ approximated by $g^{k-\frac{1}{2}}-g^{k}+g^{k-1}$). With diminishing step sizes $\color{DarkGoldenRod}{\alpha_{k} = O(1/k^{\gamma})}$, we obtain <span style="color:DarkGoldenRod">strong convergence</span> $\lim \limits_{k\to \infty}x^{k} = x^{*} \in \textrm{crit}(\psi)$ almost surely in <span style="color:DarkGoldenRod">nonconvex-nonconcave case</span>, and we have,

$$
||x^{k}-x^{*}||=
\left\{\begin{matrix}
\begin{aligned}
&O(1/\sqrt{k}), \ \ \ \ \ \ \ \ \   \ \  \ \   \   \ \   \ \ \ \ \ \ \ \ \   \ \  \ \   \   \ \  \   \ \  \ \textrm{if}\  \theta \in [0,1/2],\\
&O(1/k^{p}), p\in (0,1/2), \ \ \ \ \ \ \ \ \   \ \  \ \   \   \ \  \textrm{if}\  \theta \in (1/2,1),\\

\end{aligned}
\end{matrix}\right.
\tag{12}
$$

where $\theta \in [0,1)$ is the <span style="color:DarkGoldenRod">KL exponent</span>. Similar strong convergence results can also be obtained and proved in the Normal Map-based Extragradient Algorithm in accordance with the Supermartingale Convergence Theorem and KL inequality, whereby we show that $\\{x^{k}\\}_k$ is convergent.

**(Bonus) Useful Lemma:** In comparison to the natural residual $F_{nat}^{\lambda}(x)$, the normal map $F_{nor}^{\lambda}(z)$ can be employed as a tighter merit (gap) function in the convergence analysis for nonconvex cases, which also helps to convert the expectation of natural residual  $\mathbb{E}[F_{nat}^{\lambda}(x)]$ into a better-solved equation when analyzing stochastic algorithms.

$$
\begin{aligned}

||F_{nat}^{\lambda}(x)|| &= ||\textrm{prox}_{\lambda \varphi} (z) - \textrm{prox}_{\lambda \varphi}(x-\lambda \nabla f(x))||,\\
&\leq ||z-x+\lambda \nabla f(x)||, \ \ \  \ \color{DarkGoldenRod}{\textrm{(Non-expansiveness)}}\\
& = \lambda ||F_{nor}^{\lambda}(z)||,
\end{aligned}
\tag{13}
$$

where the natural residual $F_{nat}^{\lambda}(x)\leq \epsilon$ can be ensured after $k$ iterations under $ \lambda F_{nor}^{\lambda}(z)\leq \epsilon$, when $x=\textrm{prox}_{\lambda \varphi} (z)$.

**Our Contribution:** The main contributions of this research project can be summarized as follows.

1. <span style="color:DarkGoldenRod">First</span> stochastic algorithm to resolve the <span style="color:DarkGoldenRod">biasedness</span> of stochastic methods in <span style="color:DarkGoldenRod">nonsmooth</span> or constrained case.

2. <span style="color:DarkGoldenRod">First</span> strong convergence result for stochastic proximal-type algorithm in <span style="color:DarkGoldenRod">nonconvex-nonconcave</span> case.

3. The rate $O(1/\sqrt{k})$ matches that of OGDA in <span style="color:DarkGoldenRod">strongly convex</span> setting. Additionally, the rate $O(1/\sqrt{k})$ is <span style="color:DarkGoldenRod">tight</span>.

**Current Progress:** The paper is quite long and technical, so we still need time to check some technical details including proofs and conduct more experimemts. Articles are expected to submit into some top journals or conferences in 2023 Fall.

+ **Yilin Gu**, Hanfeng Zeng (2023). Normal Map-Based Proximal Extragradient Methods for Nonsmooth Nonconvex-Nonconcave Minmax Problem.
+ Junwen Qiu, **Yilin Gu**, Andre Milzarek (2023). A Revised Foward-Backward-Foward Method for Nonsmooth Nonconvex-Nonconcave Minmax Problem.

**(Bonus) Useful Lemma:** In comparison to the natural residual $F_{nat}^{\lambda}(x)$, the normal map $F_{nor}^{\lambda}(z)$ can be employed as a tighter merit (gap) function in the convergence analysis for nonconvex cases, which also helps to convert the expectation of natural residual  $\mathbb{E}[F_{nat}^{\lambda}(x)]$ into a better-solved equation when analyzing stochastic algorithms.

$$
\begin{aligned}

||F_{nat}^{\lambda}(x)|| &= ||\textrm{prox}_{\lambda \varphi} (z) - \textrm{prox}_{\lambda \varphi}(x-\lambda \nabla f(x))||,\\
&\leq ||z-x+\lambda \nabla f(x)||, \ \ \  \ \color{DarkGoldenRod}{\textrm{(Non-expansiveness)}}\\
& = \lambda ||F_{nor}^{\lambda}(z)||,
\end{aligned}
\tag{13}
$$

where the natural residual $F_{nat}^{\lambda}(x)\leq \epsilon$ can be ensured after $k$ iterations under $ \lambda F_{nor}^{\lambda}(z)\leq \epsilon$, when $x=\textrm{prox}_{\lambda \varphi} (z)$.



The normal map $F_{nor}^{\lambda}(z)$ helps to convert the expectation of natural residual $\mathbb{E}[F_{nat}^{\lambda}(x)]$ into a better-solved equation when analyzing stochastic algorithms, and it can be used as a tighter merit (gap) function in convergence analysis for nonconvex case compared to the natural residual.

.


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