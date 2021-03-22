extern crate hyperloglogplus;

extern crate rand;

use std::collections::hash_map::RandomState;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::*;

use hyperloglogplus::{HyperLogLog, HyperLogLogPF, HyperLogLogPlus};

fn generate_strings(count: usize) -> Vec<String> {
    let mut rng = rand::thread_rng();

    let mut workload: Vec<String> = (0..count)
        .map(|_| format!("- {} - {} -", rng.gen::<u64>(), rng.gen::<u64>()))
        .collect();

    workload.shuffle(&mut rng);

    workload
}

fn bench_insert(c: &mut Criterion) {
    let workload = generate_strings(2000);

    macro_rules! bench_impls {
        ($testname:expr, $impl:ident, $precision:expr) => {
            c.bench_function($testname, |b| {
                b.iter(|| {
                    let mut hll: $impl<String, RandomState> =
                        $impl::new($precision, RandomState::new()).unwrap();

                    for val in &workload {
                        hll.insert(val);
                    }
                })
            });
        };
    }

    bench_impls!["hyperloglog_add_p8", HyperLogLogPF, 8];
    bench_impls!["hyperloglog_add_p14", HyperLogLogPF, 14];
    bench_impls!["hyperloglog_add_p16", HyperLogLogPF, 16];

    bench_impls!["hyperloglogplus_add_p8", HyperLogLogPlus, 8];
    bench_impls!["hyperloglogplus_add_p14", HyperLogLogPlus, 14];
    bench_impls!["hyperloglogplus_add_p16", HyperLogLogPlus, 16];
}

fn bench_count(c: &mut Criterion) {
    macro_rules! bench_impls {
        ($testname:expr, $impl:ident, $precision:expr, $count:expr) => {
            let workload = generate_strings($count);

            let mut hll: $impl<String, RandomState> =
                $impl::new($precision, RandomState::new()).unwrap();

            for val in &workload {
                hll.insert(val);
            }

            c.bench_function($testname, |b| {
                b.iter(|| {
                    let val = hll.count();
                    black_box(val);
                })
            });
        };
    }

    bench_impls!["hyperloglog_count_p8", HyperLogLogPF, 8, 20_000];
    bench_impls!["hyperloglog_count_p14", HyperLogLogPF, 14, 100_000];
    bench_impls!["hyperloglog_count_p16", HyperLogLogPF, 16, 5_000_000];

    bench_impls![
        "hyperloglogplus_count_p8_below_thres",
        HyperLogLogPlus,
        8,
        200
    ];
    bench_impls![
        "hyperloglogplus_count_p8_above_thres",
        HyperLogLogPlus,
        8,
        20_000
    ];

    bench_impls![
        "hyperloglogplus_count_p14_below_thres",
        HyperLogLogPlus,
        14,
        10_000
    ];
    bench_impls![
        "hyperloglogplus_count_p14_above_thres",
        HyperLogLogPlus,
        14,
        100_000
    ];

    bench_impls![
        "hyperloglogplus_count_p16_below_thres",
        HyperLogLogPlus,
        16,
        49_000
    ];
    bench_impls![
        "hyperloglogplus_count_p16_above_thres",
        HyperLogLogPlus,
        16,
        5_000_000
    ];
}

criterion_group!(benches, bench_insert, bench_count);

criterion_main!(benches);
