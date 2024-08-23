use nalgebra::DMatrix;
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use rand_distr::Normal;
use std::collections::{HashMap, HashSet};
use std::fs;

pub const HIDDEN_SIZE: usize = 100; // number of neurons in the hidden layer
pub const SEQ_LENGTH: usize = 25; // truncation of backpropagation through time
pub const LEARNING_RATE: f64 = 1e-1;
pub const CLAMP: f64 = 5.0;
pub const SAMPLE_SIZE: usize = 200;
pub const SAMPLE_INTERVAL: usize = 1000;

#[allow(non_camel_case_types)]
type ix = usize;

struct GateModel {
    pub parameters: [DMatrix<f64>; 3],
}

impl GateModel {
    fn new(w: DMatrix<f64>, u: DMatrix<f64>, b: DMatrix<f64>) -> Self {
        Self {
            parameters: [w, u, b],
        }
    }

    fn from_distribution(
        vocab_size: usize,
        hidden_size: usize,
        normal: &Normal<f64>,
        rng: &mut ThreadRng,
    ) -> Self {
        Self {
            parameters: [
                0.01 * DMatrix::from_distribution_generic(
                    nalgebra::Dyn(hidden_size),
                    nalgebra::Dyn(vocab_size),
                    &normal,
                    rng,
                ),
                0.01 * DMatrix::from_distribution_generic(
                    nalgebra::Dyn(hidden_size),
                    nalgebra::Dyn(hidden_size),
                    &normal,
                    rng,
                ),
                DMatrix::zeros(hidden_size, 1),
            ],
        }
    }

    fn new_memory(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            parameters: [
                DMatrix::zeros(hidden_size, vocab_size),
                DMatrix::zeros(hidden_size, hidden_size),
                DMatrix::zeros(hidden_size, 1),
            ],
        }
    }

    fn w(&self) -> &DMatrix<f64> {
        &self.parameters[0]
    }

    fn u(&self) -> &DMatrix<f64> {
        &self.parameters[1]
    }

    fn b(&self) -> &DMatrix<f64> {
        &self.parameters[2]
    }

    fn update_memory(&mut self, gradient: &GateModel) {
        for i in 0..3 {
            self.parameters[i] += gradient.parameters[i].map(|x| x * x);
        }
    }

    fn update_parameters(&mut self, learning_rate: f64, gradient: &GateModel, memory: &GateModel) {
        for i in 0..3 {
            self.parameters[i] -= learning_rate
                * gradient.parameters[i]
                    .component_mul(&memory.parameters[i].map(|x| 1.0 / f64::sqrt(x + 1e-8)));
        }
    }
}

struct Model {
    pub parameters: [GateModel; 4],
}

impl Model {
    fn new(model_f: GateModel, model_i: GateModel, model_o: GateModel, model_g: GateModel) -> Self {
        Self {
            parameters: [model_f, model_i, model_o, model_g],
        }
    }

    fn new_random(vocab_size: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        Self {
            parameters: [
                GateModel::from_distribution(vocab_size, hidden_size, &normal, &mut rng),
                GateModel::from_distribution(vocab_size, hidden_size, &normal, &mut rng),
                GateModel::from_distribution(vocab_size, hidden_size, &normal, &mut rng),
                GateModel::from_distribution(vocab_size, hidden_size, &normal, &mut rng),
            ],
        }
    }

    fn new_memory(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            parameters: [
                GateModel::new_memory(vocab_size, hidden_size),
                GateModel::new_memory(vocab_size, hidden_size),
                GateModel::new_memory(vocab_size, hidden_size),
                GateModel::new_memory(vocab_size, hidden_size),
            ],
        }
    }

    fn update_memory(&mut self, gradient: &Model) {
        for i in 0..4 {
            self.parameters[i].update_memory(&gradient.parameters[i]);
        }
    }

    fn update_parameters(&mut self, learning_rate: f64, gradient: &Model, memory: &Model) {
        for i in 0..4 {
            self.parameters[i].update_parameters(
                learning_rate,
                &gradient.parameters[i],
                &memory.parameters[i],
            );
        }
    }

    fn f(&self) -> &GateModel {
        &self.parameters[0]
    }

    fn i(&self) -> &GateModel {
        &self.parameters[1]
    }

    fn o(&self) -> &GateModel {
        &self.parameters[2]
    }

    fn g(&self) -> &GateModel {
        &self.parameters[3]
    }
}

pub fn start(path: &String) {
    /* File loading */
    let data = fs::read_to_string(path).unwrap_or_else(|_| panic!("Cannot find file {path}"));
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

    // Model weights and biases
    let mut model = Model::new_random(vocab_size, HIDDEN_SIZE);

    // Memory variables for AdaGrad
    let mut memory = Model::new_memory(vocab_size, HIDDEN_SIZE);

    let mut n = 0;
    let mut p = 0;
    let mut smooth_loss = (vocab_size as f64).ln() * SEQ_LENGTH as f64;
    let mut prev_h = DMatrix::zeros(HIDDEN_SIZE, 1);
    let mut prev_c = DMatrix::zeros(HIDDEN_SIZE, 1);

    loop {
        // truncated backpropagation through time
        if p + SEQ_LENGTH + 1 >= data.len() || n == 0 {
            // reset the hidden state
            prev_h = DMatrix::zeros(HIDDEN_SIZE, 1);
            prev_c = DMatrix::zeros(HIDDEN_SIZE, 1);
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
        if n % SAMPLE_INTERVAL == 0 {
            let sample = sample(&prev_h, &prev_c, inputs[0], SAMPLE_SIZE, vocab_size, &model);
            let txt: String = sample.iter().map(|i| ix_to_char[i]).collect();
            println!("----\n {} \n----", txt);
        }

        // Retrieve loss and gradient
        let (loss, gradient, new_prev_h, new_prev_c) =
            loss_function(inputs, targets, &prev_h, &prev_c, vocab_size, &model);
        prev_h = new_prev_h;
        prev_c = new_prev_c;
        smooth_loss = smooth_loss * 0.999 + loss * 0.001;
        if n % SAMPLE_INTERVAL == 0 {
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
    prev_c: &DMatrix<f64>,
    vocab_size: usize,
    model: &Model,
) -> (f64, Model, DMatrix<f64>, DMatrix<f64>) {
    unimplemented!()
}

fn sample(
    h: &DMatrix<f64>,
    c: &DMatrix<f64>,
    seed_letter: ix,
    n: usize,
    vocab_size: usize,
    model: &Model,
) -> Vec<ix> {
    unimplemented!()
}
