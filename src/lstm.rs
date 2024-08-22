use nalgebra::DMatrix;
use rand::prelude::SliceRandom;
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

    fn update_memory(&mut self, gradient: &Model) {
        for i in 0..4 {
            self.parameters[i].update_memory(&gradient.parameters[i]);
        }
    }

    fn update_parameters(&mut self, learning_rate: f64, gradient: &Model, memory: &Model) {
        for i in 0..4 {
            self.parameters[i].update_parameters(learning_rate, &gradient.parameters[i], &memory.parameters[i]);
        }
    }
}