use std::process::ExitCode;
use tessera_load::{Config, USAGE};

fn main() -> ExitCode {
    let Some(config) = Config::parse(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match tessera_load::run(&config) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("plain data")
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("tessera-load: {e}");
            ExitCode::FAILURE
        }
    }
}
