//! Implementations of the HyperLogLog algorithm for cardinality estimation.
//!
//! HyperLogLog is an probabilistic algorithm for estimating the number of
//! *distinct* elements (*cardinality*) of a multiset. The original algorithm,
//! described by P. Flajolet et al. in *HyperLogLog: the analysis of a
//! near-optimal cardinality estimation algorithm*, can estimate cardinalities
//! well beyond 10<sup>9</sup> with a typical accuracy of 2% while using memory
//! of 1.5 kilobytes. HyperLogLog variants can improve on those results.
//!
//! All HyperLogLog variants should implement the [`HyperLogLog`] trait.
//!
//! Current implementations:
//!
//! * [`HyperLogLogPF`]
//! * [`HyperLogLogPlus`]

#![cfg_attr(feature = "bench-units", feature(test))]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod common;
#[cfg(feature = "std")]
mod constants;
#[cfg(feature = "std")]
mod encoding;
mod hyperloglog;
#[cfg(feature = "std")]
mod hyperloglogplus;
#[cfg(not(feature = "std"))]
mod log;

use core::fmt;
use core::hash::Hash;

use std::borrow::Borrow;

// Available also with no_std.
pub use crate::hyperloglog::HyperLogLogPF;

// Available only with std.
#[cfg(feature = "std")]
pub use crate::hyperloglogplus::HyperLogLogPlus;

/// A trait that should be implemented by any HyperLogLog variant.
pub trait HyperLogLog<H: Hash + ?Sized> {
    /// Adds a new value to the multiset.
    fn add<Q>(&mut self, value: &Q)
        where
            H: Borrow<Q>,
            Q: Hash + ?Sized;
    /// Estimates the cardinality of the multiset.
    fn count(&mut self) -> f64;
}

#[derive(Debug, PartialEq)]
pub enum HyperLogLogError {
    InvalidPrecision,
    IncompatiblePrecision,
}

impl fmt::Display for HyperLogLogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HyperLogLogError::InvalidPrecision => {
                "precision is out of bounds.".fmt(f)
            },
            HyperLogLogError::IncompatiblePrecision => {
                "precisions must be equal.".fmt(f)
            },
        }
    }
}
