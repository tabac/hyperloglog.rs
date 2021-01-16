use core::hash::{BuildHasher, Hash, Hasher};
use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::common::*;
use crate::HyperLogLog;
use crate::HyperLogLogError;

/// Implements the original HyperLogLog algorithm for cardinality estimation.
///
/// This implementation is based on the original paper of P. Flajolet et al:
///
/// *HyperLogLog: the analysis of a near-optimal cardinality estimation
/// algorithm.*
///
/// - Uses 5-bit registers, packed in a 32-bit unsigned integer. Thus, every
///   six registers 2 bits are not used.
/// - Supports serialization/deserialization through `serde`.
/// - Compiles in a `no_std` environment using a global allocator.
///
/// # Examples
///
/// ```
/// use std::collections::hash_map::RandomState;
/// use hyperloglogplus::{HyperLogLog, HyperLogLogPF};
///
/// let mut hll = HyperLogLogPF::new(16, RandomState::new()).unwrap();
///
/// hll.add(&12345);
/// hll.add(&23456);
///
/// assert_eq!(hll.count().trunc() as u32, 2);
/// ```
///
/// # References
///
/// - ["HyperLogLog: the analysis of a near-optimal cardinality estimation
///   algorithm", Philippe Flajolet, Éric Fusy, Olivier Gandouet and Frédéric
///   Meunier.](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
///
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperLogLogPF<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    builder:   B,
    count:     usize,
    precision: u8,
    registers: Registers,
    phantom:   PhantomData<H>,
}

impl<H, B> HyperLogLogPF<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    // Minimum precision allowed.
    const MIN_PRECISION: u8 = 4;
    // Maximum precision allowed.
    const MAX_PRECISION: u8 = 16;

    /// Creates a new HyperLogLogPF instance.
    pub fn new(precision: u8, builder: B) -> Result<Self, HyperLogLogError> {
        // Ensure the specified precision is within bounds.
        if precision < Self::MIN_PRECISION || precision > Self::MAX_PRECISION {
            return Err(HyperLogLogError::InvalidPrecision);
        }

        // Calculate register count based on given precision.
        let count = Self::register_count(precision);

        Ok(HyperLogLogPF {
            builder:   builder,
            count:     count,
            precision: precision,
            registers: Registers::with_count(count),
            phantom:   PhantomData,
        })
    }

    /// Merges the `other` HyperLogLogPF instance into `self`.
    ///
    /// Both sketches must have the same precision.
    pub fn merge<S, T>(
        &mut self,
        other: &HyperLogLogPF<S, T>,
    ) -> Result<(), HyperLogLogError>
    where
        S: Hash + ?Sized,
        T: BuildHasher,
    {
        if self.precision != other.precision() {
            return Err(HyperLogLogError::IncompatiblePrecision);
        }

        for (i, val) in other.registers_iter().enumerate() {
            self.registers.set_greater(i, val);
        }

        Ok(())
    }

    #[inline] // Returns the precision of the HyperLogLogPF instance.
    fn precision(&self) -> u8 {
        self.precision
    }

    #[inline] // Returns an iterator to the Registers' values.
    fn registers_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.registers.iter()
    }
}

impl<H, B> HyperLogLogCommon for HyperLogLogPF<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
}

impl<H, B> HyperLogLog<H> for HyperLogLogPF<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    /// Adds a new value to the multiset.
    fn add(&mut self, value: &H) {
        // Create a new hasher.
        let mut hasher = self.builder.build_hasher();
        // Calculate the hash.
        value.hash(&mut hasher);
        // Drops the higher 32 bits.
        let mut hash: u32 = hasher.finish() as u32;

        // Calculate the register's index.
        let index: usize = (hash >> (32 - self.precision)) as usize;

        // Shift left the bits of the index.
        hash = (hash << self.precision) | (1 << (self.precision - 1));

        // Count leading zeros.
        let zeros: u32 = 1 + hash.leading_zeros();

        // Update the register with the max leading zeros counts.
        self.registers.set_greater(index, zeros);
    }

    /// Estimates the cardinality of the multiset.
    fn count(&mut self) -> f64 {
        // Calculate the raw estimate.
        let (mut raw, zeros) = (
            Self::estimate_raw_plus(self.registers.iter(), self.count),
            self.registers.zeros(),
        );

        let two32 = (1u64 << 32) as f64;

        if raw <= 2.5 * self.count as f64 && zeros != 0 {
            // Apply small range correction.
            raw = Self::linear_count(self.count, zeros);
        } else if raw > two32 / 30.0 {
            // Apply large range correction.
            raw = -1.0 * two32 * ln(1.0 - raw / two32);
        }

        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use core::hash::{BuildHasher, Hasher};
    use siphasher::sip::SipHasher13;

    struct PassThroughHasher(u64);

    impl Hasher for PassThroughHasher {
        #[inline]
        fn finish(&self) -> u64 {
            self.0
        }

        #[inline]
        fn write(&mut self, _: &[u8]) {}

        #[inline]
        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }
    }

    #[derive(Serialize, Deserialize)]
    struct PassThroughHasherBuilder;

    impl BuildHasher for PassThroughHasherBuilder {
        type Hasher = PassThroughHasher;

        fn build_hasher(&self) -> Self::Hasher {
            PassThroughHasher(0)
        }
    }

    #[derive(Serialize, Deserialize)]
    struct DefaultBuildHasher;

    impl BuildHasher for DefaultBuildHasher {
        type Hasher = SipHasher13;

        fn build_hasher(&self) -> Self::Hasher {
            SipHasher13::new()
        }
    }

    #[test]
    fn test_add() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(16, builder).unwrap();

        hll.add(&0x0000000000010fff);

        assert_eq!(hll.registers.get(1), 5);

        hll.add(&0x000000000002ffff);

        assert_eq!(hll.registers.get(2), 1);

        hll.add(&0x0000000000030000);

        assert_eq!(hll.registers.get(3), 17);

        hll.add(&0x0000000000030001);

        assert_eq!(hll.registers.get(3), 17);

        hll.add(&0x00000000ff037000);

        assert_eq!(hll.registers.get(0xff03), 2);

        hll.add(&0x00000000ff030800);

        assert_eq!(hll.registers.get(0xff03), 5);

        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(4, builder).unwrap();

        hll.add(&0x000000001fffffff);
        assert_eq!(hll.registers.get(1), 1);

        hll.add(&0x00000000ffffffff);
        assert_eq!(hll.registers.get(0xf), 1);

        hll.add(&0x0000000000ffffff);
        assert_eq!(hll.registers.get(0), 5);
    }

    #[test]
    fn test_count() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(16, builder).unwrap();

        assert_eq!(hll.count(), 0.0);

        hll.add(&0x0000000000010fff);
        hll.add(&0x0000000000020fff);
        hll.add(&0x0000000000030fff);
        hll.add(&0x0000000000040fff);
        hll.add(&0x0000000000050fff);
        hll.add(&0x0000000000050fff);

        assert_eq!(hll.count().trunc() as u64, 5);
    }

    #[test]
    fn test_merge() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(16, builder).unwrap();

        hll.add(&0x00000000000101ff);
        hll.add(&0x00000000000202ff);
        hll.add(&0x00000000000304ff);
        hll.add(&0x0000000000040fff);
        hll.add(&0x0000000000050fff);
        hll.add(&0x0000000000060fff);

        assert_eq!(hll.registers.get(1), 8);
        assert_eq!(hll.registers.get(2), 7);
        assert_eq!(hll.registers.get(3), 6);
        assert_eq!(hll.registers.get(4), 5);
        assert_eq!(hll.registers.get(5), 5);
        assert_eq!(hll.registers.get(6), 5);

        let builder = PassThroughHasherBuilder {};

        let err: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(9, builder).unwrap();

        assert_eq!(
            hll.merge(&err),
            Err(HyperLogLogError::IncompatiblePrecision)
        );

        let builder = PassThroughHasherBuilder {};

        let mut other: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(16, builder).unwrap();

        hll.add(&0x00000000000404ff);
        hll.add(&0x00000000000502ff);
        hll.add(&0x00000000000601ff);

        assert_eq!(other.merge(&hll), Ok(()));

        assert_eq!(other.count().trunc() as u64, 6);

        assert_eq!(hll.registers.get(1), 8);
        assert_eq!(hll.registers.get(2), 7);
        assert_eq!(hll.registers.get(3), 6);
        assert_eq!(hll.registers.get(4), 6);
        assert_eq!(hll.registers.get(5), 7);
        assert_eq!(hll.registers.get(6), 8);
    }

    #[test]
    fn test_serialization() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            HyperLogLogPF::new(16, builder).unwrap();

        hll.add(&0x0000000000010fff);
        hll.add(&0x0000000000020fff);
        hll.add(&0x0000000000030fff);
        hll.add(&0x0000000000040fff);
        hll.add(&0x0000000000050fff);
        hll.add(&0x0000000000050fff);

        assert_eq!(hll.count().trunc() as usize, 5);

        let serialized = serde_json::to_string(&hll).unwrap();

        let mut deserialized: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.count().trunc() as usize, 5);

        deserialized.add(&0x0000000000060fff);

        assert_eq!(deserialized.count().trunc() as usize, 6);

        let mut hll: HyperLogLogPF<u64, DefaultBuildHasher> =
            HyperLogLogPF::new(16, DefaultBuildHasher {}).unwrap();

        hll.add(&0x0000000000010fff);
        hll.add(&0x0000000000020fff);
        hll.add(&0x0000000000030fff);
        hll.add(&0x0000000000040fff);
        hll.add(&0x0000000000050fff);
        hll.add(&0x0000000000050fff);

        assert_eq!(hll.count().trunc() as usize, 5);

        let serialized = serde_json::to_string(&hll).unwrap();

        let mut deserialized: HyperLogLogPF<u64, PassThroughHasherBuilder> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.count().trunc() as usize, 5);

        deserialized.add(&0x0000000000060fff);

        assert_eq!(deserialized.count().trunc() as usize, 6);
    }

    #[cfg(feature = "bench-units")]
    mod benches {
        extern crate test;

        use super::*;
        use rand::prelude::*;
        use test::{black_box, Bencher};

        #[bench]
        fn bench_add(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
                HyperLogLogPF::new(16, builder).unwrap();

            b.iter(|| {
                for i in 0u64..1000 {
                    hll.add(&(u64::max_value() - i));
                }
            })
        }

        #[bench]
        fn bench_add_with_hash(b: &mut Bencher) {
            let mut rng = rand::thread_rng();

            let workload: Vec<String> = (0..2000)
                .map(|_| {
                    format!("- {} - {} -", rng.gen::<u64>(), rng.gen::<u64>())
                })
                .collect();

            b.iter(|| {
                let mut hll: HyperLogLogPF<&String, DefaultBuildHasher> =
                    HyperLogLogPF::new(16, DefaultBuildHasher {}).unwrap();

                for val in &workload {
                    hll.add(&val);
                }

                let val = hll.count();

                black_box(val);
            })
        }

        #[bench]
        fn bench_count(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPF<u64, PassThroughHasherBuilder> =
                HyperLogLogPF::new(16, builder).unwrap();

            for i in 0u64..10000 {
                hll.add(&(u64::max_value() - i));
            }

            b.iter(|| {
                let count = hll.count();
                black_box(count);
            })
        }
    }
}
