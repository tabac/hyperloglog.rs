# HyperLogLog

[![Build status](https://github.com/tabac/hyperloglog.rs/workflows/ci/badge.svg)](https://github.com/tabac/hyperloglog.rs/actions)
[![Crates.io](https://img.shields.io/crates/v/hyperloglogplus.svg)](https://crates.io/crates/hyperloglogplus)
[![Documentation](https://docs.rs/hyperloglogplus/badge.svg)](https://docs.rs/hyperloglogplus)

HyperLogLog is a probabilistic algorithm for estimating the number of
*distinct* elements (*cardinality*) of a multiset. Several variations of the
original algorithm, described by P. Flajolet et al., have been proposed.

The following implementations are provided:

- [HyperLogLog](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- [HyperLogLog++](https://research.google/pubs/pub40671/)


## Usage

Add to `Cargo.toml`:

```toml
[dependencies]
hyperloglogplus = "*"
```

A simple example using HyperLogLog++ implementation:

```rust
use std::collections::hash_map::RandomState;
use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};

let mut hllp: HyperLogLogPlus<u64, RandomState> =
    HyperLogLogPlus::new(16, RandomState::new()).unwrap();

hllp.add(&12345);
hllp.add(&23456);

assert_eq!(hllp.count().trunc() as u32, 2);

```

## Evaluation

[Here](results/) you can find figures and discussion on experimental evaluation.
