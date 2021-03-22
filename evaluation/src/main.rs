use std::fmt;
use std::fs::File;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str;

use clap::{App, Arg, ArgMatches};
use rand::distributions::{Distribution, Standard};
use rand::prelude::*;
use rayon::prelude::*;

use hyperloglogplus::{HyperLogLog, HyperLogLogPF, HyperLogLogPlus};

struct PassThroughHasher(u64);

impl Hasher for PassThroughHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {}

    fn write_u32(&mut self, i: u32) {
        self.0 = u64::from(i);
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

struct PassThroughHasherBuilder;

impl BuildHasher for PassThroughHasherBuilder {
    type Hasher = PassThroughHasher;

    fn build_hasher(&self) -> Self::Hasher {
        PassThroughHasher(0)
    }
}

struct Estimation(u64, u64);

impl fmt::Display for Estimation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, self.1)
    }
}

// Generates random 32/64bit hash values and saves them in files.
fn generate<T>(args: &ArgMatches)
where
    T: fmt::Display,
    Standard: Distribution<T>,
{
    let (size, count, runs, output) = (
        args.value_of("size").unwrap(),
        args.value_of("count").unwrap().parse::<usize>().unwrap(),
        args.value_of("runs").unwrap().parse::<usize>().unwrap(),
        args.value_of("output").unwrap(),
    );

    (0..runs).into_par_iter().for_each(|r| {
        let values = (0..count).map(|_| rand::random::<T>()).collect();

        let filename = format!("hashes-s{}-r{}.dat", size, r);

        save(&values, filename.as_str(), output);
    });
}

macro_rules! run_impls {
    ($ident:ident, $impl:ident, $ty:ident, $sz:expr) => {
        // Runs evaluation experiments.
        fn $ident(args: &ArgMatches) {
            let (mode, precision, output) = (
                args.value_of("mode").unwrap(),
                args.value_of("precision").unwrap().parse::<u8>().unwrap(),
                args.value_of("output").unwrap(),
            );

            match mode {
                "hashes" => {
                    let files: Vec<&str> =
                        args.values_of("input").unwrap().collect();

                    files.par_iter().for_each(|file| {
                        let mut hll: $impl<$ty, _> =
                            $impl::new(precision, PassThroughHasherBuilder {})
                                .unwrap();

                        let hashes = load::<$ty>(file);

                        let estimations = hashes
                            .iter()
                            .enumerate()
                            .map(|(i, num)| {
                                hll.insert(&num);

                                Estimation(
                                    (i + 1) as u64,
                                    hll.count().trunc() as u64,
                                )
                            })
                            .collect();

                        let basename = Path::new(file)
                            .file_name()
                            .unwrap()
                            .to_str()
                            .unwrap();

                        let filename =
                            format!("est-p{}-{}", precision, basename);

                        save(&estimations, filename.as_str(), output);
                    });
                },
                "cardinalities" => {
                    let runs = args
                        .value_of("runs")
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    let cardinalities = args.value_of("cardinalities").unwrap();

                    (0..runs).into_par_iter().for_each(|r| {
                        let mut hll: $impl<$ty, _> =
                            $impl::new(precision, PassThroughHasherBuilder {})
                                .unwrap();

                        let cardinalities = load::<usize>(cardinalities);

                        let mut rng = rand::thread_rng();

                        let mut c = 0;

                        let estimations = cardinalities
                            .iter()
                            .map(|cardinality| {
                                while c < *cardinality {
                                    let num = rng.gen::<$ty>();

                                    hll.insert(&num);

                                    c += 1;
                                }

                                Estimation(c as u64, hll.count().trunc() as u64)
                            })
                            .collect();

                        let filename = format!(
                            "est-p{}-cards-s{}-r{}.dat",
                            precision, $sz, r
                        );

                        save(&estimations, filename.as_str(), output);
                    });
                },
                _ => {},
            }
        }
    };
}

// Loads the hashes saved in `filepath`.
fn load<T>(filepath: &str) -> Vec<T>
where
    T: str::FromStr + fmt::Debug,
{
    let reader = BufReader::new(File::open(filepath).unwrap());

    let mut nums = Vec::with_capacity(10000);

    for line in reader.lines() {
        nums.push(
            line.unwrap()
                .parse::<T>()
                .map_err(|_| "Parsing line failed")
                .unwrap(),
        );
    }

    nums
}

// Saves the `values` into a file with `filename` in `location`.
fn save<T>(values: &Vec<T>, filename: &str, output: &str)
where
    T: fmt::Display,
{
    let mut writer =
        BufWriter::new(File::create(Path::new(output).join(filename)).unwrap());

    for val in values {
        write!(writer, "{}\n", val).unwrap();
    }

    writer.flush().unwrap();
}

run_impls![run, HyperLogLogPF, u32, "32"];
run_impls![run_plus, HyperLogLogPlus, u64, "64"];

fn main() {
    let gen_app = App::new("gen")
        .about("generate random hash values saved in files.")
        .arg(
            Arg::with_name("size")
                .short('s')
                .long("size")
                .required(true)
                .takes_value(true)
                .possible_values(&["32", "64"]),
        )
        .arg(
            Arg::with_name("count")
                .short('c')
                .long("count")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("runs")
                .short('r')
                .long("runs")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .required(true)
                .takes_value(true),
        );

    let run_app = App::new("run")
        .about("run hyperloglog evaluation experiments.")
        .arg(
            Arg::with_name("type")
                .short('t')
                .long("type")
                .required(true)
                .takes_value(true)
                .possible_values(&["hll", "hllp"]),
        )
        .arg(
            Arg::with_name("mode")
                .short('m')
                .long("mode")
                .required(true)
                .takes_value(true)
                .possible_values(&["hashes", "cardinalities"]),
        )
        .arg(
            Arg::with_name("precision")
                .short('p')
                .long("precision")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("input")
                .multiple(true)
                .short('i')
                .long("input")
                .takes_value(true)
                .required_if("mode", "hashes"),
        )
        .arg(
            Arg::with_name("runs")
                .short('r')
                .long("runs")
                .takes_value(true)
                .required_if("mode", "cardinalities"),
        )
        .arg(
            Arg::with_name("cardinalities")
                .short('c')
                .long("cardinalities")
                .takes_value(true)
                .required_if("mode", "cardinalities"),
        )
        .arg(
            Arg::with_name("output")
                .short('o')
                .long("output")
                .required(true)
                .takes_value(true),
        );

    let matches: ArgMatches = App::new("evl")
        .about("run hyperloglog evaluation experiments")
        .arg(
            Arg::with_name("jobs")
                .short('j')
                .long("jobs")
                .takes_value(true),
        )
        .subcommand(gen_app)
        .subcommand(run_app)
        .get_matches();

    let jobs = matches
        .value_of("jobs")
        .unwrap_or("1")
        .parse::<usize>()
        .unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build_global()
        .unwrap();

    match matches.subcommand() {
        ("gen", Some(sub_matches)) => match sub_matches.value_of("size") {
            Some("32") => generate::<u32>(sub_matches),
            Some("64") => generate::<u64>(sub_matches),
            _ => {},
        },
        ("run", Some(sub_matches)) => match sub_matches.value_of("type") {
            Some("hll") => run(sub_matches),
            Some("hllp") => run_plus(sub_matches),
            _ => {},
        },

        _ => {},
    }
}
