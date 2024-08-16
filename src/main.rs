use nalgebra::DMatrix;
use std::collections::HashSet;
use std::fs;

struct Model {
    wxh: DMatrix<f64>,
    whh: DMatrix<f64>,
    why: DMatrix<f64>,
    bh: DMatrix<f64>,
    by: DMatrix<f64>,
}

struct Gradient {
    dwxh: DMatrix<f64>,
    dwhh: DMatrix<f64>,
    dwhy: DMatrix<f64>,
    dbh: DMatrix<f64>,
    dby: DMatrix<f64>,
}

fn main() {
    /* File loading */
    let file_path = "data/shakespeare-40k.txt";
    let data =
        fs::read_to_string(file_path).expect(format!("Cannot find file {file_path}").as_str());
    let chars: HashSet<char> = data.chars().collect();
    let vocab_size = chars.len();

    println!("Data has {} characters, {} unique", data.len(), chars.len());

    /* Hyperparameters */
    let hidden_size = 100; // number of neurons in the hidden layer
    let seq_length = 25; // truncation of backpropagation through time

    /* Model parameters */
    let wxh = 0.01 * DMatrix::new_random(hidden_size, vocab_size);
    let whh = 0.01 * DMatrix::new_random(hidden_size, hidden_size);
    let why = 0.01 * DMatrix::new_random(vocab_size, hidden_size);
    //let bh = DMatrix::zeros(hidden_size, 1);
    //let by = DMatrix::zeros(vocab_size, 1);
}

fn loss_function(
    inputs: Vec<usize>,
    targets: Vec<usize>,
    prev_h: DMatrix<f64>,
    vocab_size: usize,
    model: Model,
) -> (f64, Gradient, DMatrix<f64>) {
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

        hs.push(
            (model.wxh.clone() * xs[t].clone()
                + model.whh.clone() * hs[t].clone()
                + model.bh.clone())
            .map(f64::tanh),
        ); // note that `hs` is shifted from one time step because `hs[0]` is the previous hidden state
        ys.push(model.why.clone() * hs[t + 1].clone() + model.by.clone());
        ps.push(ys[t].map(f64::exp).exp() / ys[t].map(f64::exp).sum()); // probabilities for next chars (softmax of ys)

        loss = loss - ps[t][(targets[t], 0)].ln(); // cross-entropy loss
    }

    /* Backward pass */
    let mut dwxh = DMatrix::zeros(model.wxh.nrows(), model.wxh.ncols());
    let mut dwhh = DMatrix::zeros(model.whh.nrows(), model.whh.ncols());
    let mut dwhy = DMatrix::zeros(model.why.nrows(), model.why.ncols());
    let mut dbh = DMatrix::zeros(model.bh.nrows(), model.bh.ncols());
    let mut dby = DMatrix::zeros(model.by.nrows(), model.by.ncols());

    let mut dnext_h = DMatrix::zeros(hs[0].nrows(), hs[0].ncols());

    for t in (0..inputs.len()).rev() {
        let mut dy = ps[t].clone();
        dy[targets[t]] = dy[targets[t]] - 1.0;
        dwhy += dy.clone() * hs[t + 1].transpose();
        dby += dy.clone();

        let dh = model.why.transpose() * dy + dnext_h.clone();
        let dhraw = (DMatrix::from_element(hs[t + 1].nrows(), hs[t + 1].ncols(), 1.0)
            - hs[t + 1].clone() * hs[t + 1].clone())
            * dh; // tanh

        dbh += dhraw.clone();
        dwxh += dhraw.clone() * xs[t].transpose();
        dwhh += dhraw.clone() * hs[t].transpose();
        dnext_h = model.whh.transpose() * dhraw;
    }

    // Gradient cliping
    for dparam in [&mut dwxh, &mut dwhh, &mut dwhy, &mut dbh, &mut dby] {
        let norm = dparam.norm();
        if norm > 5.0 {
            *dparam *= 5.0 / norm
        }
    }

    let grad = Gradient {
        dwxh,
        dwhh,
        dwhy,
        dbh,
        dby,
    };

    return (loss, grad, hs[inputs.len()].clone());
}
