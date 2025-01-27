\section{Derivation of the ELBO loss for TVPS}
Here we show how to derive the \textit{evidence lower bound (ELBO)} of our proposed model in details. We started from the marginal likelihood of observable variables $\bm(x,y)$

\begin{align*}
\log p(\bm{x,y}) &= \mathbb{E}_{p(\bm{z, \omega}|\bm{x,y})} [\log p(\bm{x,y})] \\
&= \mathbb{E}_{p(\bm{z,\omega}|\bm{x,y})} \Bigg[ \log \Bigg[ \frac{ p(\bm{x,y},\bm{z,\omega})}{p(\bm{z,\omega}|\bm{x,y})} \Bigg] \Bigg] + \text{KL}(p(\bm{z,\omega}|\bm{x,y}) || p(\bm{x,y},\bm{z,\omega}))
\end{align*}
where $\bm{x,y}$ are data samples of input and output. $\bm{z}$ denotes the latent variables, the posterior of which is defined to be mixture of Gaussian. The mixture of Gaussian has $K$ Gaussian components and the mixing coefficients $\bm{s}=\{s_{1},\cdots,s_{K}\}$. We denote the corresponding latent random variable to determine which Gaussian component to draw sample from as $\omega$. 

As discussed in the main text, we parameterize the posterior $p(\bm{z,\omega}|\bm{x})$ of latent variables $\bm{z},\omega$ using a neural encoder $F_{\phi}$. We denote the posterior as $q_{\phi}(\bm{z, \omega}|\bm{x})$. To decode representations sampled from $\bm{z}$, we design a decoder $G_{\theta}$ which takes a random sample and output a reconstruction $\bm{y}$, with additional information from the encoder. Therefore, unlike regular VAE, our generative model is a composition of both $F_{\phi}$ and $G_{\theta}$. We train $F_{\phi}$ and $G_{\theta}$ by maximizing the marginal likelihood of observable variables $(\bm{x,y})$.
\begin{align*}
\log p(\bm{x,y}) &= \mathbb{E}_{q_{\phi}(\bm{z, \omega}|\bm{x})} [\log p_{\theta,\phi}(\bm{x,y})] \\
&= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \log \Bigg[ \frac{ p_{\theta,\phi}(\bm{x,y},\bm{z,\omega})}{q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg] \Bigg] + \text{KL}(q_{\phi}(\bm{z,\omega}|\bm{x}) || p_{\theta,\phi}(\bm{x,y},\bm{z,\omega}))
\end{align*}


The ELBO thus reads:
\begin{align*}
    \mathcal{L}_{\theta,\phi}(\bm{x},\bm{y})
    &= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \log \Bigg[ \frac{ p_{\theta,\phi}(\bm{x,y},\bm{z,\omega})}{q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg] \Bigg] \\
    &= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \log \Bigg[ \frac{ p_{\theta,\phi}(\bm{x,y},\bm{z,\omega})}{q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg] \Bigg] \\
    &= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[ \log p_{\theta,\phi}(\bm{x,y},\bm{z,\omega}) - \log q_{\phi}(\bm{z,\omega}|\bm{x}) \big] \\
    &= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[ \log p_{\theta,\phi}(\bm{x,y}|\bm{z}) + \log p(\bm{z}|\bm{\omega}) + \log p(\bm{\omega}) - \log q_{\phi}(\bm{z,\omega}|\bm{x}) \big] \\
    &= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[ \log p_{\theta,\phi}(\bm{x,y}|\bm{z}) \big] + \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[\log p(\bm{z},\bm{\omega}) - \log q_{\phi}(\bm{z,\omega}|\bm{x}) \big] \\
    &= \underbrace{\mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[ \log p_{\theta,\phi}(\bm{x,y}|\bm{z}) \big]}_{\text{reconstruction loss}} + \underbrace{\text{KL}(q_{\phi}(\bm{z,\omega}|\bm{x}) || p(\bm{z},\bm{\omega}))}_{\text{regularization of latent distributions}} \\
    &= \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[ \log p_{\theta,\phi}(\bm{x,y}|\bm{z}) \big] + \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[\log p(\bm{z},\bm{\omega}) \big] - \mathbb{E}_{q_{\phi}(\bm{z,\omega}|\bm{x})} \big[ \log q_{\phi}(\bm{z,\omega}|\bm{x}) \big] \\
    &=\mathcal{L}_{logxy|z\omega} + \mathcal{L}_{logz\omega} - \mathcal{L}_{logz\omega|x}
 \end{align*}

The ELBO consists of three terms, where the first one is a log-likelihood loss which measures the reconstruction error, the second and third ones correspond to the priors and posterior of the latent variables $\bm{z}$ and $\bm{\omega}$. 
\begin{align*}
\mathcal{L}_{logxy|z\omega} 
= \mathbb{E}_{\bm{z}, \omega} \big[ \log p_{\theta,\phi}(\bm{x,y}|\bm{z}) \big]  \approx \frac{1}{N} \sum_{n=1}^{N} \log p_{\theta,\phi}(\bm{x,y}|\bm{\mathring{z}_{n}, \mathring{\omega}_{n}}) 
\end{align*}
where $\{ \mathring{\bm{z}}_{n}, \mathring{\omega}_{n}, n=1,\cdots, N\}$ are random samples drawn from $\mathcal{N}(0,\bm{I})$ and $\mathcal{M}(K)$. \\

We assume the prior of $z$ is a standard Gaussian distribution $\mathcal{N}(0,\bm{I})$.
% , the prior for $\omega$ is a multinominal distribution with $s_{1}=s_{2}=\cdots=s_{K}=1/K$.
\begin{align*}
\mathcal{L}_{logz\omega} 
&= \mathbb{E}_{\bm{z}, \omega \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Big[\log p(\bm{z},\bm{\omega}) \Big]  \\
&= \mathbb{E}_{\bm{\omega} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[\int_{\bm{z}} q_{\phi}(\bm{z,\omega}|\bm{x}) \log p(\bm{z},\bm{\omega}) d\bm{z} \Bigg]  \\
&= \mathbb{E}_{\bm{\omega} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \int_{\bm{z}} \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k}  \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \log \Big ( \frac{1}{(2\pi)^{D/2}} \bm{s}_{0}^{\omega_k} \exp  \Big\{ -\frac{1}{2} \sum_{i} z_{ki}^2 \Big\} \Big) d\bm{z} \Bigg] \\
&= \mathbb{E}_{\bm{\omega} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \int_{\bm{z}} \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k}  \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \Big[ \log \Big ( \frac{1}{(2\pi)^{D/2}} \bm{s}_{0}^{\omega_k} \Big ) -\frac{1}{2} \sum_{i} z_{ki}^2 \Big] d\bm{z} \Bigg] \\
&= \mathbb{E}_{\bm{\omega} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \underbrace{\frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \log \Big( \frac{1}{(2\pi)^{D/2}} \bm{s}_{0}^{\omega_k} \Big) \int_{\bm{z}} \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2}\Big\} d\bm{z}}_{Part 1}  \\
&+ \underbrace{\frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \int_{\bm{z}} \Big( -\frac{1}{2} \sum_{i} z_{ki}^2 \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \Big) d\bm{z}}_{Part 2} \Bigg]
\end{align*}

To simplify \textit{the integral terms} in Part1 and Part2, we can obtain:

Part 1:

\begin{align*}
\int_{\bm{z}} \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} d\bm{z}
&= (\prod_{i}^{D} \sigma_{ki}) \int_{\bm{z}} \exp \Big\{ -\frac{1}{2} \sum_{i} \varepsilon_{ki}^2 \Big\} d\varepsilon_{k1}d\varepsilon_{k2}\cdots\varepsilon_{kD} \\
&= (\prod_{i}^{D} \sigma_{ki}) (2\pi)^{D/2}
\end{align*}

Part 2:

We will decompose the D-dimensional summation into the form of the \textit{D-th} term plus a \textit{D-1} dimensional summation, and solve it recursively.

\begin{align*}
\int_{\bm{z}} \Big( -\frac{1}{2} \sum_{i} z_{ki}^2 \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \Big) d\bm{z}
&=\int_{\bm{z}} -\frac{1}{2} z_{kD}^{2}\exp \Big\{ (-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} dz_{kD}dz_{k(D-1)}\cdots dz_{k1} \\
&+\int_{\bm{z}} -\frac{1}{2} \sum_{i}^{D-1} z_{ki}^{2}\exp \Big\{ (-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} dz_{kD}dz_{k(D-1)}\cdots dz_{k1}
\end{align*}

It can be derived that:

\begin{align}
&\int_{\bm{z}} \sigma_{kD}^{2}\varepsilon_{kD}^{2}\exp\Big\{-\frac{1}{2}\varepsilon_{kD}^2\Big\}d\varepsilon_{kD} = (2\pi)^{1/2}\sigma_{kD}^{2} \\
&\int_{\bm{z}} \mu_{kD}^{2} \exp\Big\{-\frac{1}{2}\varepsilon_{kD}^2\Big\}d\varepsilon_{kD} = (2\pi)^{1/2}\mu_{kD}^{2} \\
&\int_{\bm{z}} 2\mu_{kD}\sigma_{kD}\varepsilon_{kD} \exp\Big\{-\frac{1}{2}\varepsilon_{kD}^2\Big\}d\varepsilon_{kD} = -2\mu_{kD}\sigma_{kD}\int_{\bm{z}} d\exp \Big\{-\frac{1}{2}\varepsilon_{kD}^2 \Big\} = 0
\end{align}

Then the \textit{D-th} term can be expressed as:

\begin{align}
&\int_{\bm{z}} -\frac{1}{2} z_{kD}^{2}\exp \Big\{ (-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} dz_{kD}dz_{k(D-1)}\cdots dz_{k1} \\
&= \int_{\bm{z}} \exp \Big\{ (-\frac{1}{2}\sum_{i}^{D-1}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} (-\frac{1}{2} z_{kD}^{2}) \exp \Big\{ (-\frac{1}{2} \frac{(z_{kD}-\mu_{kD})^{2}}{\sigma_{kD}^{2}})\Big\}dz_{kD}dz_{k(D-1)}\cdots dz_{k1} \\
&= \int_{\bm{z}} \exp \Big\{ (-\frac{1}{2}\sum_{i}^{D-1}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} (-\frac{\sigma_{kD}}{2}) (\sigma_{kD} \varepsilon_{kD} + \mu_{kD})^{2} \exp \Big\{ -\frac{1}{2} \varepsilon_{kD}^2 \Big\}d\varepsilon_{kD}dz_{k(D-1)}\cdots dz_{k1} \\
&= -\frac{\sigma_{kD}}{2}(2\pi)^{1/2}(\mu_{kD}^{2}+\sigma_{kD}^{2})\int_{\bm{z}} \exp \Big\{ (-\frac{1}{2}\sum_{i}^{D-1}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\}dz_{k(D-1)}\cdots dz_{k1} \\
&= -\frac{\sigma_{kD}}{2}(2\pi)^{1/2}(\mu_{kD}^{2}+\sigma_{kD}^{2}) \Big ( \prod_{i=1}^{D-1} \sigma_{ki} \Big ) (2\pi)^{(D-1)/2} \\
&=-\frac{1}{2}(\sigma_{kD}^{2}+\mu_{kD}^{2})\Big ( \prod_{i=1}^{D} \sigma_{ki} \Big ) (2\pi)^{D/2}
\end{align}

And the \textit{D-1} dimensional summation can be expressed as:

\begin{align}
&\int_{\bm{z}} -\frac{1}{2} \sum_{i}^{D-1} z_{ki}^{2}\exp \Big\{ (-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} dz_{kD}dz_{k(D-1)}\cdots dz_{k1} \\
&= \int_{\bm{z}} -\frac{1}{2} \sum_{i}^{D-1} z_{ki}^{2}\exp \Big\{ (-\frac{1}{2}\sum_{i}^{D-1}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} \exp \Big\{ (-\frac{1}{2} \frac{(z_{kD}-\mu_{kD})^{2}}{\sigma_{kD}^{2}})\Big\}dz_{kD}dz_{k(D-1)}\cdots dz_{k1} \\
&= \sigma_{kD}(2\pi)^{1/2}\int_{\bm{z}} -\frac{1}{2} \sum_{i}^{D-1} z_{ki}^{2}\exp \Big\{ (-\frac{1}{2}\sum_{i}^{D-1}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})\Big\} dz_{k(D-1)}\cdots dz_{k1}
\end{align}

Letting $x_D = \int_{\bm{z}} \Big( -\frac{1}{2} \sum_{i} z_{ki}^2 \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \Big) dz_{kD}dz_{k(D-1)}\cdots dz_{k1}$, we have:
\begin{align}
x_D 
&= -\frac{1}{2}(\sigma_{kD}^{2}+\mu_{kD}^{2})\Big ( \prod_{i=1}^{D} \sigma_{ki} \Big ) (2\pi)^{D/2} + \sigma_{kD}(2\pi)^{1/2} x_{D-1} \\
\intertext{It can be observed that $x_D$ forms a geometric sequence,then substituting $D = 1$ and applying the formula for the sum of a geometric series, we can obtain:} \\
x_1
&= -\frac{1}{2}(\sigma_{k1}^2+\mu_{k1}^2)\sigma_{k1}(2\pi)^{1/2} \\
x_D
&= -\frac{1}{2}\Big ( \prod_{i=1}^{D}\sigma_{ki}\Big )(2\pi)^{D/2} \Big [ \sum_{i}^{D} \sigma_{ki}^2 + \sum_{i}^{D} \mu_{ki}^2 \Big ]
\end{align}

Therefore, we can obtain:
\begin{align}
\mathcal{L}_{logz\omega} 
&= \mathbb{E}_{\bm{\omega} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \Big(\bm{s}_{k}^{\omega_k} \log \Big( \frac{1}{(2\pi)^{D/2}} \bm{s}_{0}^{\omega_k} \Big) - \frac{1}{2} \bm{s}_{k}^{\omega_k} \Big( \sum_{i}^{D} \sigma_{ki}^{2} + \sum_{i}^{D} \mu_{ki}^{2} \Big) \Big) \Bigg ]\\
&= \mathbb{E}_{\bm{\omega} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} \Bigg[ \bm{s}_{k}^{\omega_k} \Big(\log \frac{1}{K} - \frac{D}{2} \log 2\pi  - \frac{1}{2}  \sum_{i}^{D} \sigma_{ki}^{2} - \frac{1}{2}  \sum_{i}^{D} \mu_{ki}^{2} \Big) \Bigg ]
\end{align}

where $\bm{\omega} = \{\omega_{1},\cdots,\omega_{K}\}$ is an one-hot vector indicating which Gaussian component is activated. 

Given that the posterior $q_{\phi}(\bm{z,\omega}|\bm{x})$ is assumed to be Gaussian mixture, we can calculate the second term as follows
\begin{align}
&\mathcal{L}_{logz\omega|x} \\
&= \mathbb{E}_{\bm{z},\bm{\omega}} \Big[\log q_{\phi}(\bm{z,\omega}|\bm{x}) \Big]  \\
&= \mathbb{E}_{\bm{\omega}} \Bigg[\int_{\bm{z} \sim q_{\phi}(\bm{z,\omega}|\bm{x})} q_{\phi}(\bm{z,\omega}|\bm{x}) \log q_{\phi}(\bm{z,\omega}|\bm{x}) d\bm{z} \Bigg]  \\
&= \mathbb{E}_{\bm{\omega}} \Bigg[ \int_{\bm{z}} \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k}  \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \log \Big ( \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k}  \exp \Big\{ -\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2}  \Big\} \Big) d\bm{z} \Bigg] \\
&= \mathbb{E}_{\bm{\omega}} \Bigg[ \int_{\bm{z}} \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k}  \exp \Big\{-\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \Big( \log \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k}  (-\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2}) \Big) d\bm{z} \Bigg] \\
&= \mathbb{E}_{\bm{\omega}} \Bigg[ \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \Big[ \int_{\bm{z}} \log \left( \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \right) \exp \Big\{-\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} d\bm{z} \\
&- \int_{\bm{z}} \frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \exp \Big\{-\frac{1}{2} \sum_{i} \frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2} \Big\} \Big] d\bm{z} \Bigg] \\
&= \mathbb{E}_{\bm{\omega}} \Bigg[ \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \Big[ \log \left( \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \right) \left( \prod_{i}^{D} \sigma_{ki} \right) (2\pi)^{D/2} - \frac{D}{2} \left( \prod_{i}^{D} \sigma_{ki} \right) (2\pi)^{D/2} \Big] \Bigg]
\\ 
&= \mathbb{E}_{\bm{\omega}} \Bigg[ \bm{s}_{k}^{\omega_k} \Big[ \log \left( \frac{1}{(2\pi)^{D/2}} \frac{1}{\prod_{i}^{D} \sigma_{ki}} \bm{s}_{k}^{\omega_k} \right) - \frac{D}{2} \Big] \Bigg]
\\ 
&= \mathbb{E}_{\bm{\omega} \sim \mathcal{M}(K)} \Bigg[ \bm{s}_{k}^{\omega_k} \Bigg( - \frac{D}{2} \log (2\pi)  - \frac{D}{2} - {\sum_{i}^{D} \log \sigma_{ki}} + \log \bm{s}_{k}^{\omega_k} \Bigg) \Bigg] \\
\intertext{Below we show the key aspects of the ELBO derivation:}\\
&-\frac{1}{2}\int_{-\infty}^{+\infty}\sum_{i}z_{ki}^{2} \exp \Big \{(-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}) \Big\} dz_{k1} \\
&=-\frac{1}{2}\Bigg[\underbrace{\int_{-\infty}^{+\infty}z_{k1}^{2}e^{(-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}dz_{k1}}_{A}
+\underbrace{\int_{-\infty}^{+\infty}\sum_{i=2}^{D} z_{ki}^{2}e^{(-\frac{1}{2}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}dz_{k1}}_{B}\Bigg]
\end{align}
Solving for A and B separately yields:
\begin{align}
A &= \int_{-\infty}^{+\infty}-\sigma_{k1}^2z_{k1}de^{(-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}\\
&= -\sigma_{k1}^2[z_{k1}e^{(-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}|_{-\infty}^{+\infty}-\int_{-\infty}^{+\infty}e^{(-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}dz_{k1}]\\ 
&= -\sigma_{k1}^2[0-\int_{-\infty}^{+\infty}e^{(-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}dz_{k1}]\\ 
&= \sigma_{k1}^2\int_{-\infty}^{+\infty}e^{-\frac{1}{2}\frac{(z_{k1}-\mu_{k1})^{2}}{\sigma_{k1}^{2}}}e^{-\frac{1}{2}\sum_{i=2}^{D}\frac{(z_{ki}-\mu_{ki})^2}{\sigma_{ki}^2}}dz_{k1}\\ 
&= \sigma_{k1}^2\sigma_{k1}\sqrt{2\pi}e^{-\frac{1}{2}(\sum_{i=2}^{D}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}\\ 
&= \sigma_{k_1}^3\sqrt{2\pi}e^{-\frac{1}{2}\sum_{i=2}^{D}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}}\\
\intertext{Similarly we can derive:}
B&=\sum_{i=2}^{D} z_{ki}^{2}\sigma_{k1}\sqrt{2\pi}e^{(-\frac{1}{2}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}
\end{align}

Let:

\begin{align}
x_{D}&=\int_{-\infty}^{+\infty}-\frac{1}{2}\sum_{i}z_{ki}^{2}e^{(-\frac{1}{2}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}dz\\
\intertext{So we can get:}
-\frac{1}{2}\left[A + B \right] 
& =-\frac{1}{2}\left[\sigma_{k_1}^3\sqrt{2\pi}e^{-\frac{1}{2}\sum_{i=2}^{D}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}} + \sum_{i=2}^{D} z_{ki}^{2}\sigma_{k1}\sqrt{2\pi}e^{(-\frac{1}{2}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}\right]\\  
& =-\frac{1}{2}\left[\sigma_{k1}^3\sqrt{2\pi}\sigma_{k2}\sigma_{k3}\cdots\sigma_{kD}(\sqrt{2\pi})^{D-1}+\sigma_{k1}\sqrt{2\pi}\sum_{i=2}^{D}z_{ki}^2e^{(-\frac{1}{2}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}})}\right]\\  
& =-\frac{1}{2}\left[\sigma_{k1}^2\prod_{i=1}^{D}\sigma_{ki}(2\pi)^{\frac{D}{2}}+\sigma_{k1}\sqrt{2\pi}x_{D-1}\right]\\  
x_D &=- \frac{1}{2}[\sigma_{kD}^2\prod_{i=1}^{D}\sigma_{ki}(2\pi)^{\frac{D}{2}}]+\sigma_{kD}\sqrt{2\pi}x_{D-1}\\  
x_1 &= - \frac{1}{2}\sigma_1^3\sqrt{2\pi}\\  
x_0 &= -\frac{1}{2}(\sqrt{2\pi})^{D}\prod_{i=1}^{D}\sigma_{ki}(\sum_{i=1}^{D}\sigma_{ki}^2)
\end{align}
At this point, the equation can be converted into a problem of summing a series.
\begin{align} 
&\int_{z}\frac{1}{(2\pi)^{\frac{D}{2}}}s_{k}^{T_{k}}e^{-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}}[\log(\frac{1}{(2\pi)^{\frac{D}{2}}}s_{k}^{T_{k}})-\frac{1}{2}\sum_{i=1}^2z_{ki}^2]dz \\
&=\frac{1}{(2\pi)^{\frac{D}{2}}}s_{k}^{T_{k}}[\underbrace{\int_{z}e^{-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}}\log(\frac{1}{(2\pi)^{\frac{D}{2}}}s_{k}^{T_{k}})dz}_{C}+\underbrace{\int_{z}-\frac{1}{2}\sum_{i=1}^{D}z_{ki}^2e^{-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}}dz}_{D}]
\end{align}

\begin{align} 
C &= \log(\frac{1}{(2\pi)^{\frac{D}{2}}}S_{k}^{T_{k}})\int_{z}e^{-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}}dz=\log(\frac{1}{(2\pi)^{\frac{D}{2}}}S_{k}^{T_{k}})\prod_{i=1}^{D}\sigma_{ki}(2\pi)^{\frac{D}{2}}\\  
D &= \int_{z}-\frac{1}{2}\sum_{i=1}^{D}z_{ki}^2e^{-\frac{1}{2}\sum_{i}\frac{(z_{ki}-\mu_{ki})^{2}}{\sigma_{ki}^{2}}}dz=-\frac{1}{2}(2\pi)^{\frac{D}{2}}\prod_{i=1}^{D}\sigma_{ki}(\sum_{i=1}^{D}\sigma_{ki}^2)\\  
\end{align}
Therefore, the original expression can be expressed as:
\begin{align} 
S_{k}^{Tk}\log(\frac{1}{(2\pi)^{\frac{D}{2}}}S_{k}^{T_{k}})\prod_{i=1}^{D}\sigma_{ki}-\frac{1}{2}S_{k}^{T_{k}}\prod_{i=1}^{D}\sigma_{ki}(\sum_{i=1}^{D}\sigma_{ki}^2)
\end{align}


\twocolumn
\section{Detailed diagnostic opinions from professional ophthalmologists.}

Herein, we provide detailed clinical diagnostic opinions from professional ophthalmologists on the samples generated by the proposed TVPS in part. 

Column 1: In the real fundus image, the optic disc color is normal, with clear boundaries. Yellow-white exudates are visible beside the macula and in the temporal side.

In the generated vessels, vascular leakage is observed in the inferotemporal side of the optic disc, with irregular vessel shapes.

In the generated fundus image, yellow-white exudations are visible in the inferotemporal side of the optic disc. 

In the generated fundus image used for training, yellow-white exudations are visible in the temporal side and inferotemporal side of the optic disc.

Column 3: In the real fundus image, the optic disc color is normal, with clear boundaries. Yellow-white exudations, hemorrhages, and microaneurysms are visible beside the macula and in the peripheral retina.

In the generated vessels, vascular leakage is visible in the inferotemporal side of the optic disc. The shape of some vessels is obscured by hemorrhages, leading to an irregular course.

In the generated fundus image, yellow-white exudations are visible in the temporal side and inferotemporal side of the optic disc, along with patchy hemorrhages and microaneurysms.

In the generated fundus image used for training, yellow-white exudations and multiple hemorrhages are visible in the temporal side and inferotemporal side of the optic disc.

Column 4: In the real fundus image, the optic disc color is normal, with clear boundaries. Hemorrhages and exudations are visible in the posterior pole.

In the generated vessels, there is visible vascular deficiency in the superotemporal side of the optic disc, which appears to be due to vascular obscuration caused by hemorrhage.

In the generated fundus image, a small amount of preretinal hemorrhage and exudation is visible in the superotemporal side.

In the generated fundus image used for training, yellow-white exudations and multiple hemorrhages are visible in the temporal side and superotemporal side of the optic disc.

Column 6: In the real fundus image, the optic disc color is normal, with clear boundaries. The vessels in the superotemporal side have a tortuous shape, and flame-shaped hemorrhages are visible.

In the generated vessels, hemorrhage in the superotemporal side of the optic disc leads to obscuration and results in discontinuity of some vessels.

In the generated fundus image, hemorrhages and exudations are visible in the superotemporal side of the optic disc.

In the generated fundus image used for training, hemorrhages and exudations are visible in the superotemporal side of the optic disc, with a larger extent compared to the generated fundus images.

Column 7: In the real fundus image, the optic disc color is normal. The veins have a tortuous course, and flame-shaped hemorrhages are visible all around.

In the generated vessels, the vessels all around are obscured by hemorrhages, resulting in an incomplete course, especially severe in the superior nasal area.

In the generated fundus image, the optic disc boundaries are indistinct, with exudations visible around. The tortuous veins and a small amount of flame-shaped hemorrhages are visible all around.

In the generated fundus image used for training, flame-shaped hemorrhages are visible all around, and exudations are visible on the nasal side of the optic disc.

Column 8: In the real fundus image, the optic disc color is normal. A preretinal hemorrhage of approximately 8 PD in size is visible in the superotemporal side, and small patchy flame-shaped hemorrhages are visible in the inferotemporal side.

In the generated vessels, retinal hemorrhages in the inferotemporal side obscure the vessels, and at the same time, some vessels have an incomplete shape.

In the generated fundus image, the boundaries of the optic disc are visible. In the posterior pole, preretinal and subretinal hemorrhages of approximately 8 PD in size are visible, and exudations are visible in the periphery.

In the generated fundus image used for training, the optic disc color is normal. A preretinal hemorrhage of approximately 8 PD in size is visible in the superotemporal side, and small patchy exudations are visible in the inferotemporal side.

Column 10: In the real fundus image, the optic disc color is normal, with clear boundaries. A scar-like lesion is visible in the macula, accompanied by a small amount of hemorrhage in the center.

In the generated vessels, vascular deficiency is visible in the inferotemporal side of the optic disc, seemingly due to vascular obscuration caused by hemorrhage.

In the generated fundus image, a scar is visible in the macular area, and hemorrhage is visible in the inferotemporal side of the scar.

In the generated fundus image used for training, a scar is visible in the macular area, and hemorrhage is visible in the inferotemporal side of the scar.

Column 11: In the real fundus image, the optic disc color is normal, with clear boundaries. Large areas of exudation are visible in the temporal side of the macula, along with preretinal and subretinal hemorrhages.

In the generated vessels, vascular leakage is visible in the superotemporal side of the optic disc. The course of some vessels is obscured by hemorrhages, leading to an irregular shape, or in other words, there is partial occlusion of the vessels in the superotemporal side.

In the generated fundus image, yellow-white exudations are visible in the temporal side of the optic disc, and subretinal hemorrhage is visible in the temporal side of the macula.

In the generated fundus image used for training, multiple yellow-white exudations are visible in the temporal side and inferotemporal side of the optic disc, and preretinal and subretinal hemorrhages are visible in the temporal side of the macula.

Column 12: In the real fundus image, the optic disc color is normal. Hemorrhages are visible in the macular area, and exudations are visible around the macula.

In the generated vessels, the shape of the vessels in the temporal side of the optic disc is irregular.

In the generated fundus image, the boundaries of the optic disc are visible. A scar of approximately 2/3*2/3 PD in size is visible in the macular area, accompanied by surrounding hemorrhages.

In the generated fundus image used for training, the optic disc color is normal. Hemorrhages are visible in the macular area, accompanied by surrounding exudations, and a scar is visible on the nasal side of the macula.


\section{Details of the network architecture}

Here, we present the specific architectures of the encoder-decoder and discriminator utilized in TVPS.

\renewcommand{\thefigure}{S1}
\begin{figure}[!htp]
  \centering
  \includegraphics[width=0.4\textwidth]{TVPS/Figs/network_details/network_details_v3.pdf}
  \caption{Network architectures of the \textit{neural encoder $F_{\phi}$ and decoder $G_{\theta}$} (a) and \textit{adversarial discriminator $D_{\psi}$} (b) in the proposed TVPS.\textcircled{\tiny U} stands for feature concatenation.}
    \label{fig_network_details}
\end{figure}

