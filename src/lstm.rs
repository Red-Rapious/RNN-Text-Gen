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

    fn zeros(vocab_size: usize, hidden_size: usize) -> Self {
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

    fn w_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.parameters[0]
    }

    fn u_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.parameters[1]
    }

    fn b_mut(&mut self) -> &mut DMatrix<f64> {
        &mut self.parameters[2]
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

    fn clip(&mut self) {
        for i in 0..3 {
            self.parameters[i] = self.parameters[i].map(|x| x.clamp(-CLAMP, CLAMP))
        }
    }
}

struct Model {
    pub parameters: [GateModel; 4],
    pub wy: DMatrix<f64>,
    pub by: DMatrix<f64>,
}

impl Model {
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
            wy: 0.01
                * DMatrix::from_distribution_generic(
                    nalgebra::Dyn(vocab_size),
                    nalgebra::Dyn(hidden_size),
                    &normal,
                    &mut rng,
                ),
            by: DMatrix::zeros(vocab_size, 1),
        }
    }

    fn zeros(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            parameters: [
                GateModel::zeros(vocab_size, hidden_size),
                GateModel::zeros(vocab_size, hidden_size),
                GateModel::zeros(vocab_size, hidden_size),
                GateModel::zeros(vocab_size, hidden_size),
            ],
            wy: DMatrix::zeros(vocab_size, hidden_size),
            by: DMatrix::zeros(vocab_size, 1),
        }
    }

    fn update_memory(&mut self, gradient: &Model) {
        for i in 0..4 {
            self.parameters[i].update_memory(&gradient.parameters[i]);
        }
        self.wy += gradient.wy.map(|x| x * x);
        self.by += gradient.by.map(|x| x * x);
    }

    fn update_parameters(&mut self, learning_rate: f64, gradient: &Model, memory: &Model) {
        for i in 0..4 {
            self.parameters[i].update_parameters(
                learning_rate,
                &gradient.parameters[i],
                &memory.parameters[i],
            );
        }
        self.wy -= learning_rate
            * gradient
                .wy
                .component_div(&memory.wy.map(|x| (x + 1e-8).sqrt()));
        self.by -= learning_rate
            * gradient
                .by
                .component_div(&memory.by.map(|x| (x + 1e-8).sqrt()));
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

    fn f_mut(&mut self) -> &mut GateModel {
        &mut self.parameters[0]
    }

    fn i_mut(&mut self) -> &mut GateModel {
        &mut self.parameters[1]
    }

    fn o_mut(&mut self) -> &mut GateModel {
        &mut self.parameters[2]
    }

    fn g_mut(&mut self) -> &mut GateModel {
        &mut self.parameters[3]
    }

    fn clip(&mut self) {
        for i in 0..4 {
            self.parameters[i].clip()
        }
        self.wy = self.wy.map(|x| x.clamp(-CLAMP, CLAMP));
        self.by = self.by.map(|x| x.clamp(-CLAMP, CLAMP));
    }
}

fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
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
    let mut memory = Model::zeros(vocab_size, HIDDEN_SIZE);

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
    // input vectors
    let mut xs = Vec::with_capacity(inputs.len());
    // hidden states
    let mut hs = Vec::with_capacity(inputs.len());
    // cell states
    let mut cs = Vec::with_capacity(inputs.len());
    // output vectors (probabilities before softmax)
    let mut ys = Vec::with_capacity(inputs.len());
    // char probabilities (softmax(y))
    let mut ps = Vec::with_capacity(inputs.len());
    // input gates
    let mut is = Vec::with_capacity(inputs.len());
    // forget gates
    let mut fs = Vec::with_capacity(inputs.len());
    // output gates
    let mut os = Vec::with_capacity(inputs.len());
    // cell input gates
    let mut gs = Vec::with_capacity(inputs.len());

    // note that `hs` and `cs` are shifted from one time step because `hs[0]` is the previous hidden state
    hs.push(prev_h.clone());
    cs.push(prev_c.clone());

    let mut loss = 0.0;

    // Forward pass
    for t in 0..inputs.len() {
        xs.push(DMatrix::zeros(vocab_size, 1));
        xs[t][inputs[t]] = 1.0;

        is.push((model.i().w() * &xs[t] + model.i().u() * &hs[t] + model.i().b()).map(sigmoid));
        fs.push((model.f().w() * &xs[t] + model.f().u() * &hs[t] + model.f().b()).map(sigmoid));
        os.push((model.o().w() * &xs[t] + model.o().u() * &hs[t] + model.o().b()).map(sigmoid));
        gs.push((model.g().w() * &xs[t] + model.g().u() * &hs[t] + model.g().b()).map(f64::tanh));

        cs.push(fs[t].component_mul(&cs[t]) + is[t].component_mul(&gs[t]));
        hs.push(os[t].component_mul(&cs[t + 1].map(f64::tanh)));

        ys.push(&model.wy * &hs[t + 1] + &model.by);
        ps.push(ys[t].map(f64::exp) / ys[t].map(f64::exp).sum()); // probabilities for next chars (softmax of ys)

        loss -= ps[t][targets[t]].ln(); // cross-entropy loss
    }

    // Backward pass
    let mut grad = Model::zeros(vocab_size, HIDDEN_SIZE);
    let mut dnext_h = DMatrix::zeros(hs[0].nrows(), hs[0].ncols());
    let mut dnext_c = DMatrix::zeros(hs[0].nrows(), hs[0].ncols());

    for t in (0..inputs.len()).rev() {
        // probabilities
        let mut dy = ps[t].clone();
        dy[targets[t]] -= 1.0;

        grad.wy += &dy * hs[t + 1].transpose();
        grad.by += &dy;

        let dh = model.wy.transpose() * dy + &dnext_h;

        // output gate
        let do_ = dh.component_mul(&cs[t + 1].map(f64::tanh));
        let do_raw = do_.component_mul(&os[t].map(|x| x * (1.0 - x)));
        *grad.o_mut().w_mut() += &do_raw * xs[t].transpose();
        *grad.o_mut().u_mut() += &do_raw * hs[t].transpose();
        *grad.o_mut().b_mut() += &do_raw;

        let dc = dh
            .component_mul(&os[t])
            .component_mul(&cs[t + 1].map(|x| 1.0 - x.tanh() * x.tanh()))
            + &dnext_c;

        // cell input gate
        let dg = dc.component_mul(&is[t]);
        let dc_raw = dg.component_mul(&gs[t].map(|x| 1.0 - x * x));
        *grad.g_mut().w_mut() += &dc_raw * xs[t].transpose();
        *grad.g_mut().u_mut() += &dc_raw * hs[t].transpose();
        *grad.g_mut().b_mut() += &dc_raw;

        // input gate
        let di = dc.component_mul(&gs[t]);
        let di_raw = di.component_mul(&is[t].map(|x| x * (1.0 - x)));
        *grad.i_mut().w_mut() += &di_raw * xs[t].transpose();
        *grad.i_mut().u_mut() += &di_raw * hs[t].transpose();
        *grad.i_mut().b_mut() += &di_raw;

        // forget gate
        let df = dc.component_mul(&cs[t]);
        let df_raw = df.component_mul(&fs[t].map(|x| x * (1.0 - x)));
        *grad.f_mut().w_mut() += &df_raw * xs[t].transpose();
        *grad.f_mut().u_mut() += &df_raw * hs[t].transpose();
        *grad.f_mut().b_mut() += &df_raw;

        dnext_h = model.f().u().transpose() * &df_raw
            + model.i().u().transpose() * &df_raw
            + model.o().u().transpose() * &df_raw
            + model.g().u().transpose() * &df_raw;
        dnext_c = fs[t].component_mul(&dc);
    }

    grad.clip();

    (
        loss,
        grad,
        hs[inputs.len()].clone(),
        cs[inputs.len()].clone(),
    )
}

fn sample(
    h: &DMatrix<f64>,
    c: &DMatrix<f64>,
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

    let mut h = h.clone();
    let mut c = c.clone();

    for _ in 0..n {
        // feedforward pass
        let i = (model.i().w() * &x + model.i().u() * &h + model.i().b()).map(sigmoid);
        let f = (model.f().w() * &x + model.f().u() * &h + model.f().b()).map(sigmoid);
        let o = (model.o().w() * &x + model.o().u() * &h + model.o().b()).map(sigmoid);
        let g = (model.g().w() * &x + model.g().u() * &h + model.g().b()).map(f64::tanh);

        c = f.component_mul(&c) + i.component_mul(&g);
        h = o.component_mul(&c.map(f64::tanh));

        // output
        let y = &model.wy * &h + &model.by;
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
