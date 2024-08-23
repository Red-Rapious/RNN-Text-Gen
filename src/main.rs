use std::env;

mod lstm;
mod rnn;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("usage: rnn-text-gen <architecture> <filepath>\n\narchitectures:\n    rnn \tvanilla Recurrent Neural Network\n    lstm\tLong Short-Term Memory");
        std::process::exit(1);
    }
    let architecture = &args[1];
    let path = &args[2];

    match architecture.as_str() {
        "rnn" => rnn::start(path),
        "lstm" => lstm::start(path),
        _ => {
            println!("usage: rnn-text-gen <architecture> <filepath>\n\narchitectures:\n    rnn \tvanilla Recurrent Neural Network\n    lstm\tLong Short-Term Memory");
            std::process::exit(1);
        }
    }

    rnn::start(path);
}
