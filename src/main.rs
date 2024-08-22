use std::env;

mod rnn;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        println!("usage: rnn-text-gen <filepath>");
        std::process::exit(1);
    }
    let path = &args[1];

    rnn::start(path);
}