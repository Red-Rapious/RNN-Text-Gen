use nalgebra::DMatrix;
use rand::prelude::SliceRandom;
use rand_distr::Normal;
use std::collections::{HashMap, HashSet};
use std::fs;

pub const FILE_PATH: &'static str = "data/shakespeare-40k.txt";
pub const HIDDEN_SIZE: usize = 100; // number of neurons in the hidden layer
pub const SEQ_LENGTH: usize = 25; // truncation of backpropagation through time
pub const LEARNING_RATE: f64 = 1e-1;
pub const CLAMP: f64 = 5.0;
pub const SAMPLE_SIZE: usize = 200;

#[allow(non_camel_case_types)]
type ix = usize;

struct Model {
    pub parameters: [DMatrix<f64>; 5],
}

impl Model {
    fn new(
        wxh: DMatrix<f64>,
        whh: DMatrix<f64>,
        why: DMatrix<f64>,
        bh: DMatrix<f64>,
        by: DMatrix<f64>,
    ) -> Self {
        Self {
            parameters: [wxh, whh, why, bh, by],
        }
    }

    fn wxh(&self) -> &DMatrix<f64> {
        &self.parameters[0]
    }

    fn whh(&self) -> &DMatrix<f64> {
        &self.parameters[1]
    }

    fn why(&self) -> &DMatrix<f64> {
        &self.parameters[2]
    }

    fn bh(&self) -> &DMatrix<f64> {
        &self.parameters[3]
    }

    fn by(&self) -> &DMatrix<f64> {
        &self.parameters[4]
    }

    fn update_memory(&mut self, gradient: &Model) {
        for i in 0..5 {
            self.parameters[i] += gradient.parameters[i].map(|x| x * x);
        }
    }

    fn update_parameters(&mut self, learning_rate: f64, gradient: &Model, memory: &Model) {
        for i in 0..5 {
            self.parameters[i] -= learning_rate
                * gradient.parameters[i]
                    .component_mul(&memory.parameters[i].map(|x| 1.0 / f64::sqrt(x + 1e-8)));
        }
    }
}

fn main() {
    /* File loading */
    let data =
        fs::read_to_string(FILE_PATH).unwrap_or_else(|_| panic!("Cannot find file {FILE_PATH}"));
    let chars: HashSet<char> = data.chars().collect();
    let ix_to_char: HashMap<ix, char> = chars.clone().into_iter().enumerate().collect();
    let char_to_ix: HashMap<char, ix> = chars
        .clone()
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();
    let vocab_size = chars.len();

    println!("Data has {} characters, {} unique", data.len(), chars.len());

    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    // Model parameters
    let mut model = Model::new(
        0.01 * DMatrix::from_distribution_generic(nalgebra::Dyn(HIDDEN_SIZE), nalgebra::Dyn(vocab_size), &normal, &mut rng),
        0.01 * DMatrix::from_distribution_generic(nalgebra::Dyn(HIDDEN_SIZE), nalgebra::Dyn(HIDDEN_SIZE), &normal, &mut rng),
        0.01 * DMatrix::from_distribution_generic(nalgebra::Dyn(vocab_size), nalgebra::Dyn(HIDDEN_SIZE), &normal, &mut rng),
        DMatrix::zeros(HIDDEN_SIZE, 1),
        DMatrix::zeros(vocab_size, 1),
    );

    // Memory variables for AdaGrad
    let mut memory = Model::new(
        DMatrix::zeros(model.wxh().nrows(), model.wxh().ncols()),
        DMatrix::zeros(model.whh().nrows(), model.whh().ncols()),
        DMatrix::zeros(model.why().nrows(), model.why().ncols()),
        DMatrix::zeros(model.bh().nrows(), model.bh().ncols()),
        DMatrix::zeros(model.by().nrows(), model.by().ncols()),
    );

    let mut n = 0;
    let mut p = 0;
    let smooth_loss = (vocab_size as f64).ln() * SEQ_LENGTH as f64;
    let mut prev_h = DMatrix::zeros(HIDDEN_SIZE, 1);

    loop {
        // truncated backpropagation through time
        if p + SEQ_LENGTH + 1 >= data.len() || n == 0 {
            // reset the hidden state
            prev_h = DMatrix::zeros(HIDDEN_SIZE, 1);
            p = 0;
        }

        let inputs: Vec<ix> = data[p..(p + SEQ_LENGTH)]
            .chars()
            .map(|c| char_to_ix[&c])
            .collect();
        let targets: Vec<ix> = data[(p + 1)..(p + SEQ_LENGTH + 1)]
            .chars()
            .map(|c| char_to_ix[&c])
            .collect();

        // sample from the model
        if n % 100 == 0 {
            let sample = sample(&prev_h, inputs[0], SAMPLE_SIZE, vocab_size, &model);
            let txt: String = sample.iter().map(|i| ix_to_char[i]).collect();
            println!("----\n {} \n----", txt);
        }

        // Retrieve loss and gradient
        let (loss, gradient, new_prev_h) =
            loss_function(inputs, targets, &prev_h, vocab_size, &model);
        prev_h = new_prev_h;
        let smooth_loss = smooth_loss * 0.999 + loss * 0.001;
        if n % 100 == 0 {
            println!("iteration {}, loss: {}", n, smooth_loss);
        }

        // Update parameters with AdaGrad
        memory.update_memory(&gradient);
        model.update_parameters(LEARNING_RATE, &gradient, &memory);

        p += SEQ_LENGTH;
        n += 1;
    }
}

fn loss_function(
    inputs: Vec<usize>,
    targets: Vec<usize>,
    prev_h: &DMatrix<f64>,
    vocab_size: usize,
    model: &Model,
) -> (f64, Model, DMatrix<f64>) {
    let mut xs = Vec::with_capacity(inputs.len());
    let mut hs = Vec::with_capacity(inputs.len());
    let mut ys = Vec::with_capacity(inputs.len());
    let mut ps = Vec::with_capacity(inputs.len());

    hs.push(prev_h.clone());
    let mut loss = 0.0;

    /* Forward pass */
    for t in 0..inputs.len() {
        xs.push(DMatrix::zeros(vocab_size, 1));
        xs[t][inputs[t]] = 1.0;

        hs.push((model.wxh() * &xs[t] + model.whh() * &hs[t] + model.bh()).map(f64::tanh)); // note that `hs` is shifted from one time step because `hs[0]` is the previous hidden state
        ys.push(model.why() * &hs[t + 1] + model.by());
        ps.push(ys[t].map(f64::exp) / ys[t].map(f64::exp).sum()); // probabilities for next chars (softmax of ys)

        loss -= ps[t][targets[t]].ln(); // cross-entropy loss
    }

    /* Backward pass */
    let mut dwxh = DMatrix::zeros(model.wxh().nrows(), model.wxh().ncols());
    let mut dwhh = DMatrix::zeros(model.whh().nrows(), model.whh().ncols());
    let mut dwhy = DMatrix::zeros(model.why().nrows(), model.why().ncols());
    let mut dbh = DMatrix::zeros(model.bh().nrows(), model.bh().ncols());
    let mut dby = DMatrix::zeros(model.by().nrows(), model.by().ncols());

    let mut dnext_h = DMatrix::zeros(hs[0].nrows(), hs[0].ncols());

    for t in (0..inputs.len()).rev() {
        let mut dy = ps[t].clone();
        dy[targets[t]] -= 1.0;
        dwhy += &dy * hs[t + 1].transpose();
        dby += &dy;

        let dh = model.why().transpose() * dy + &dnext_h;

        let dhraw = hs[t + 1].map(|x| 1.0 - x * x).component_mul(&dh); // tanh

        dbh += &dhraw;
        dwxh += &dhraw * xs[t].transpose();
        dwhh += &dhraw * hs[t].transpose();
        dnext_h = model.whh().transpose() * dhraw;
    }

    // Gradient cliping
    for dparam in [&mut dwxh, &mut dwhh, &mut dwhy, &mut dbh, &mut dby] {
        *dparam = dparam.map(|x| f64::clamp(x, -CLAMP, CLAMP));
    }

    let grad = Model::new(dwxh, dwhh, dwhy, dbh, dby);

    (loss, grad, hs[inputs.len()].clone())
}

fn sample(
    h: &DMatrix<f64>,
    seed_letter: ix,
    n: usize,
    vocab_size: usize,
    model: &Model,
) -> Vec<ix> {
    // input vector
    let mut x = DMatrix::zeros(vocab_size, 1);
    x[seed_letter] = 1.0;
    // generated letters
    let mut generated_ixes: Vec<ix> = Vec::with_capacity(n);

    let mut rng = rand::thread_rng();

    for _ in 0..n {
        // feedforward pass
        let h = (model.wxh() * x + model.whh() * h + model.bh()).map(f64::tanh);

        // output
        let y = model.why() * h + model.by();
        // apply softmax to obtain probabilities
        let p = y.map(f64::exp) / y.map(f64::exp).sum();

        // randomly sample the next character using the distribution p
        let ixes: Vec<ix> = (0..vocab_size).collect();
        let ix = *ixes.choose_weighted(&mut rng, |i| p[*i]).unwrap();
        generated_ixes.push(ix);

        // update the next input
        x = DMatrix::zeros(vocab_size, 1);
        x[ix] = 1.0;
    }

    generated_ixes
}
