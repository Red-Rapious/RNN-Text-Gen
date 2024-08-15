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
) {
    let mut xs = Vec::with_capacity(inputs.len());
    let mut hs = Vec::with_capacity(inputs.len());
    //let mut ys = Vec::with_capacity(inputs.len());
    //let mut ps = Vec::with_capacity(inputs.len());

    hs.push(prev_h.clone());
    let loss = 0;

    /* Forward pass */
    for t in 0..inputs.len() {
        xs.push(DMatrix::zeros(vocab_size, 1));
        xs[t][inputs[t]] = 1.0;

        hs.push((model.wxh.clone() * xs[t].clone() + model.whh.clone() * hs[t].clone() + model.bh.clone()).map(f64::tanh)); // note that `hs` is shifted from one time step because `hs[0]` is the previous hidden state
    }

    /* Backward pass */
}
