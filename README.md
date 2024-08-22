# RNN-Text-Gen
A small language model using a Recurrent Neural Network to generate text based on a corpus.

## Execution
The implementation is done is Rust and based on [this blog post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). To launch it, simply run:
```bash
cargo run --release
```
Hyperparameters are defined in [`src/main.rs`](src/main.rs); you can for instance change the path of the file used for training, which defaults to `"data/shakespeare-40k.txt"`.

## Theory
This simple RNN used the following recurring equations:
$$
\begin{aligned}
h_t &= \tanh\left(W_{hh}h_{t-1}+W_{xh}x_t + b_h\right)\\
y_t &= W_{hy}h_t+b_y
\end{aligned}
$$

The optimization is done with AdaGrad. For each parameter $\theta_{t,i}$:
$$
\begin{aligned}
        m_{t+1,i} &= m_{t,i} + \nabla\mathcal{L}(\theta_t)_i^2\\
        \theta_{t+1,i} &= \theta_{t,i} - \frac{\eta}{\sqrt{m_{t+1,i}+\varepsilon}}\cdot\nabla\mathcal{L}(\theta_t)_i
\end{aligned}
$$
where $\varepsilon=10^{-8}$ and $\eta$ is the learning rate, $0.1$ in the code.