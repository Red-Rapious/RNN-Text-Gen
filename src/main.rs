use std::env;

mod lstm;
mod rnn;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("usage: rnn-text-gen <architecture> <filepath>\n\narchitectures:\n    rnn \tvanilla Recurrent Neural Network\n    lstm\tLong Short-Term Memory");
        std::process::exit(1);
    }
    let path = &args[1];

    rnn::start(path);
}
