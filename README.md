# RNN-Text-Gen
A small language model using a Recurrent Neural Network to generate text based on a corpus.

## Execution
To launch the code, run the following command:
```bash
cargo run --release <architecture> <filepath>
```
The `architecture` provided must be equal to `rnn` (for a vanilla Recurrent Neural Network) or `lstm` (for a Long Short-Term Memory architecture). The parameter passed in `<filepath>` is used as a training text for the network. A showcase file containing 40k lines of Shakespeare's plays is provided in `"data/shakespeare-40k.txt"`.

The implementation is done is Rust and uses the [`nalgebra`](https://nalgebra.org/) library for linear algebra computations. The vanilla RNN implementation is based on [this blog post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), that I adapted to support LSTM.

Hyperparameters are defined in [`src/rnn.rs`](src/rnn.rs); you can for instance change the number of neurons in the hidden layer.

## Theory
### Vanilla RNN
The simple RNN uses the following recurring equations:
```math
\begin{align*}
h_t &= \tanh\left(W_{hh}h_{t-1}+W_{xh}x_t + b_h\right)\\
y_t &= W_{hy}h_t+b_y
\end{align*}
```

### LSTM
The Long Short-Term Memory architecture uses the following recurring equations:
```math
\begin{align*}
    \begin{bmatrix}
        i\\
        f\\
        o\\
        g
    \end{bmatrix}
    &= \begin{bmatrix}
        \sigma\\
        \sigma\\
        \sigma\\
        \tanh
    \end{bmatrix} W \begin{bmatrix}
        h_{t-1}\\
        x_t
    \end{bmatrix} \\
    c_t &= f \odot c_{t-1} + i\odot g\\
    h_t &= o \odot \tanh(c_t)
\end{align*}
```
where $\odot$ denotes the Hadamard (component-wise) product, `.component_mul` in the Rust code.

### Optimization (AdaGrad)
The optimization is done with AdaGrad. For each parameter $\theta_{t,i}$:
```math
\begin{align*}
        m_{t+1,i} &= m_{t,i} + \nabla\mathcal{L}(\theta_t)_i^2\\
        \theta_{t+1,i} &= \theta_{t,i} - \frac{\eta}{\sqrt{m_{t+1,i}+\varepsilon}}\cdot\nabla\mathcal{L}(\theta_t)_i
\end{align*}
```
where $\varepsilon=10^{-8}$ and $\eta$ is the learning rate, $0.1$ in the code.