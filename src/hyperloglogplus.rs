use std::collections::HashSet;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::common::*;
use crate::constants;
use crate::encoding::DifIntVec;
use crate::HyperLogLog;
use crate::HyperLogLogError;

/// Implements the HyperLogLog++ algorithm for cardinality estimation.
///
/// This implementation is based on the paper:
///
/// *HyperLogLog in Practice: Algorithmic Engineering of a State of The Art
/// Cardinality Estimation Algorithm.*
///
/// The empirical data used for bias estimation are those provided by Google
/// and can be found [here](http://goo.gl/iU8Ig).
///
///
/// # Examples
///
/// ```
/// use std::collections::hash_map::RandomState;
/// use hyperloglogplus::{HyperLogLog, HyperLogLogPlus};
///
/// let mut hllp: HyperLogLogPlus<u64, RandomState> =
///     HyperLogLogPlus::new(16, RandomState::new()).unwrap();
///
/// hllp.add(&12345);
/// hllp.add(&23456);
///
/// assert_eq!(hllp.count().trunc() as u32, 2);
/// ```
///
#[derive(Serialize, Deserialize, Debug)]
pub struct HyperLogLogPlus<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    builder:   B,
    precision: u8,
    counts:    (usize, usize, usize),
    tmpset:    HashSet<u32>,
    sparse:    DifIntVec,
    registers: Option<RegistersPlus>,
    phantom:   PhantomData<H>,
}

impl<H, B> HyperLogLogPlus<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
    // Minimum precision allowed.
    const MIN_PRECISION: u8 = 4;
    // Maximum precision allowed.
    const MAX_PRECISION: u8 = 18;
    // Maximum precision in sparse representation.
    const PRIME_PRECISION: u8 = 25;

    /// Creates a new HyperLogLogPlus instance.
    pub fn new(precision: u8, builder: B) -> Result<Self, HyperLogLogError> {
        // Ensure the specified precision is within bounds.
        if precision < Self::MIN_PRECISION || precision > Self::MAX_PRECISION {
            return Err(HyperLogLogError::InvalidPrecision);
        }

        let count = Self::register_count(precision);
        let counts = (
            count,
            Self::register_count(Self::PRIME_PRECISION - 1),
            RegistersPlus::size_in_bytes(count),
        );

        Ok(HyperLogLogPlus {
            builder:   builder,
            precision: precision,
            counts:    counts,
            tmpset:    HashSet::new(),
            sparse:    DifIntVec::new(),
            registers: None,
            phantom:   PhantomData,
        })
    }

    /// Merges the `other` HyperLogLogPlus instance into `self`.
    ///
    /// Both sketches must have the same precision. Merge can trigger
    /// the transition from sparse to normal representation.
    pub fn merge<S, T>(
        &mut self,
        other: &HyperLogLogPlus<S, T>,
    ) -> Result<(), HyperLogLogError>
    where
        S: Hash + ?Sized,
        T: BuildHasher,
    {
        if self.precision != other.precision() {
            return Err(HyperLogLogError::IncompatiblePrecision);
        }

        if other.is_sparse() {
            if self.is_sparse() {
                // Both sketches are in sparse representation.
                //
                // Insert all the hash codes of other into `tmpset`.
                for hash_code in other.tmpset.iter() {
                    self.tmpset.insert(*hash_code);
                }
                for hash_code in other.sparse.into_iter() {
                    self.tmpset.insert(hash_code);
                }

                // Merge temporary set into sparse representation.
                if self.tmpset.len() * 100 > self.counts.2 {
                    self.merge_sparse()
                }
            } else {
                // The other sketch is in sparse representation but not self.
                //
                // Decode all the hash codes and update the self's
                // corresponding Registers.
                let registers = self.registers.as_mut().unwrap();

                for hash_code in other.tmpset.iter() {
                    let (zeros, index) = other.decode_hash(*hash_code);

                    if registers.get(index) < zeros {
                        registers.set(index, zeros);
                    }
                }

                for hash_code in other.sparse.into_iter() {
                    let (zeros, index) = other.decode_hash(hash_code);

                    if registers.get(index) < zeros {
                        registers.set(index, zeros);
                    }
                }
            }
        } else {
            if self.is_sparse() {
                // The other sketch is in normal representation but self
                // is in sparse representation.
                //
                // Turn sparse into normal.
                self.merge_sparse();

                if self.is_sparse() {
                    self.sparse_to_normal();
                }
            }

            // Merge registers from both sketches.
            let registers = self.registers.as_mut().unwrap();
            let other_registers_iter = other.registers_iter().unwrap();

            for (i, val) in other_registers_iter.enumerate() {
                if registers.get(i) < val {
                    registers.set(i, val);
                }
            }
        }

        Ok(())
    }

    #[inline] // Returns the precision of the HyperLogLogPF instance.
    fn precision(&self) -> u8 {
        self.precision
    }

    #[inline] // Returns an iterator to the Registers' values.
    fn registers_iter(&self) -> Option<impl Iterator<Item = u32> + '_> {
        self.registers
            .as_ref()
            .and_then(|registers| Some(registers.iter()))
    }

    #[inline] // Returns true if the HyperLogLog is using the
              // sparse representation.
    fn is_sparse(&self) -> bool {
        self.registers.is_none()
    }

    #[inline] // Encodes the hash value as a u32 integer.
    fn encode_hash(&self, mut hash: u64) -> u32 {
        let index: u64 = u64::extract(hash, 64, 64 - Self::PRIME_PRECISION);

        let dif: u64 =
            u64::extract(hash, 64 - self.precision, 64 - Self::PRIME_PRECISION);

        if dif == 0 {
            // Shift left the bits of the index.
            hash = (hash << Self::PRIME_PRECISION) |
                (1 << Self::PRIME_PRECISION - 1);

            // Count leading zeros.
            let zeros: u32 = 1 + hash.leading_zeros();

            return ((index as u32) << 7) | (zeros << 1) | 1;
        }

        (index << 1) as u32
    }

    #[inline] // Extracts the index from a encoded hash.
    fn index(&self, hash_code: u32) -> usize {
        if hash_code & 1 == 1 {
            return u32::extract(hash_code, 32, 32 - self.precision) as usize;
        }

        u32::extract(
            hash_code,
            Self::PRIME_PRECISION + 1,
            Self::PRIME_PRECISION - self.precision + 1,
        ) as usize
    }

    #[inline] // Decodes a hash into the number of leading zeros and
              // the index of the correspondingn hash.
    fn decode_hash(&self, hash_code: u32) -> (u32, usize) {
        if hash_code & 1 == 1 {
            return (
                u32::extract(hash_code, 7, 1) +
                    (Self::PRIME_PRECISION - self.precision) as u32,
                self.index(hash_code),
            );
        }

        let hash =
            hash_code << (32 - Self::PRIME_PRECISION + self.precision - 1);

        (hash.leading_zeros() + 1, self.index(hash_code))
    }

    // Creates a set of Registers for the given precision and copies the
    // register values from the sparse representation to the normal one.
    fn sparse_to_normal(&mut self) {
        let mut registers: RegistersPlus =
            RegistersPlus::with_count(self.counts.0);

        for hash_code in self.sparse.into_iter() {
            let (zeros, index) = self.decode_hash(hash_code);

            if zeros > registers.get(index) {
                registers.set(index, zeros);
            }
        }

        self.registers = Some(registers);

        self.tmpset.clear();

        self.sparse.clear();
    }

    // Merges the hash codes stored in the temporary set to the sparse
    // representation.
    fn merge_sparse(&mut self) {
        let mut set_codes: Vec<u32> = self.tmpset.iter().copied().collect();

        set_codes.sort();

        let mut buf = DifIntVec::with_capacity(self.sparse.len());

        let (mut set_iter, mut buf_iter) =
            (set_codes.iter(), self.sparse.into_iter());

        let (mut set_hash_option, mut buf_hash_option) =
            (set_iter.next(), buf_iter.next());

        while set_hash_option.is_some() || buf_hash_option.is_some() {
            if set_hash_option.is_none() {
                buf.push(buf_hash_option.unwrap());

                buf_hash_option = buf_iter.next();

                continue;
            }
            if buf_hash_option.is_none() {
                buf.push(*set_hash_option.unwrap());

                set_hash_option = set_iter.next();

                continue;
            }

            let (set_hash_code, buf_hash_code) =
                (*set_hash_option.unwrap(), buf_hash_option.unwrap());

            if set_hash_code == buf_hash_code {
                buf.push(set_hash_code);

                set_hash_option = set_iter.next();
                buf_hash_option = buf_iter.next();
            } else if set_hash_code > buf_hash_code {
                buf.push(buf_hash_code);

                buf_hash_option = buf_iter.next();
            } else {
                buf.push(set_hash_code);

                set_hash_option = set_iter.next();
            }
        }

        self.sparse = buf;

        self.tmpset.clear();

        if self.sparse.len() > self.counts.2 {
            self.sparse_to_normal();
        }
    }

    // Returns an estimated bias correction based on empirical data.
    fn estimate_bias(&self, raw: f64) -> f64 {
        // Get a reference to raw estimates/biases for precision.
        let biases = &constants::BIAS_DATA
            [(self.precision - Self::MIN_PRECISION) as usize];
        let estimates = &constants::RAW_ESTIMATE_DATA
            [(self.precision - Self::MIN_PRECISION) as usize];

        // Raw estimate is first/last in estimates. Return the first/last bias.
        if raw <= estimates[0] {
            return biases[0];
        } else if estimates[estimates.len() - 1] <= raw {
            return biases[biases.len() - 1];
        }

        // Raw estimate is somewhere in between estimates.
        // Binary search for the calculated raw estimate.
        //
        // Here we unwrap because neither the values in `estimates`
        // nor `raw` are going to be NaN.
        let res =
            estimates.binary_search_by(|est| est.partial_cmp(&raw).unwrap());

        let (prv, idx) = match res {
            Ok(idx) => (idx - 1, idx),
            Err(idx) => (idx - 1, idx),
        };

        // Return linear interpolation between raw's neighboring points.
        let ratio = (raw - estimates[prv]) / (estimates[idx] - estimates[prv]);

        // Calculate bias.
        biases[prv] + ratio * (biases[idx] - biases[prv])
    }

    #[inline] // Returns an empirically determined threshold to decide on
              // the use of linear counting.
    fn threshold(precision: u8) -> f64 {
        match precision {
            4 => 10.0,
            5 => 10.0,
            6 => 40.0,
            7 => 80.0,
            8 => 220.0,
            9 => 400.0,
            10 => 900.0,
            11 => 1800.0,
            12 => 3100.0,
            13 => 6500.0,
            14 => 11500.0,
            15 => 20000.0,
            16 => 50000.0,
            17 => 120000.0,
            18 => 350000.0,
            _ => unreachable!(),
        }
    }
}

impl<H, B> HyperLogLogCommon for HyperLogLogPlus<H, B>
where
    H: Hash + ?Sized,
    B: BuildHasher,
{
}

impl<H, B> HyperLogLog<H> for HyperLogLogPlus<H, B>
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
        // Use a 64-bit hash value.
        let mut hash: u64 = hasher.finish();

        match &mut self.registers {
            Some(registers) => {
                // We use normal representation.

                // Calculate the register's index.
                let index: usize = (hash >> (64 - self.precision)) as usize;

                // Shift left the bits of the index.
                hash = (hash << self.precision) | (1 << (self.precision - 1));

                // Count leading zeros.
                let zeros: u32 = 1 + hash.leading_zeros();

                // Update the register with the max leading zeros counts.
                if zeros > registers.get(index) {
                    registers.set(index, zeros);
                }
            },
            None => {
                // We use sparse representation.

                // Encode hash value.
                let hash_code = self.encode_hash(hash);

                // Insert hash_code into temporary set.
                self.tmpset.insert(hash_code);

                // Merge temporary set into sparse representation.
                if self.tmpset.len() * 100 > self.counts.2 {
                    self.merge_sparse()
                }
            },
        }
    }

    /// Estimates the cardinality of the multiset.
    fn count(&mut self) -> f64 {
        // Merge tmpset into sparse representation.
        if self.registers.is_none() {
            self.merge_sparse();
        }

        match self.registers.as_mut() {
            Some(registers) => {
                // We use normal representation.

                // Calculate the raw estimate.
                let (mut raw, zeros) =
                    Self::estimate_raw(registers.iter(), self.counts.0);

                // Apply correction if required.
                if raw <= 5.0 * self.counts.0 as f64 {
                    raw -= self.estimate_bias(raw);
                }

                if zeros != 0 {
                    let correction = Self::linear_count(self.counts.0, zeros);

                    // Use linear counting only if value below threshold.
                    if correction <= Self::threshold(self.precision) {
                        raw = correction;
                    }
                }

                raw
            },
            None => {
                // We use sparse representation.

                // Calculate number of registers set to zero.
                let zeros = self.counts.1 - self.sparse.count();

                // Use linear counting to approximate.
                Self::linear_count(self.counts.1, zeros)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{BuildHasher, Hasher};

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
        type Hasher = DefaultHasher;

        fn build_hasher(&self) -> Self::Hasher {
            DefaultHasher::new()
        }
    }

    #[test]
    fn test_normal_add() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        hll.sparse_to_normal();

        assert!(hll.registers.is_some());

        hll.add(&0x00010fffffffffff);

        assert_eq!(hll.registers.as_ref().unwrap().get(1), 5);

        hll.add(&0x0002ffffffffffff);

        assert_eq!(hll.registers.as_ref().unwrap().get(2), 1);

        hll.add(&0x0003000000000000);

        assert_eq!(hll.registers.as_ref().unwrap().get(3), 49);

        hll.add(&0x0003000000000001);

        assert_eq!(hll.registers.as_ref().unwrap().get(3), 49);

        hll.add(&0xff03700000000000);

        assert_eq!(hll.registers.as_ref().unwrap().get(0xff03), 2);

        hll.add(&0xff03080000000000);

        assert_eq!(hll.registers.as_ref().unwrap().get(0xff03), 5);

        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(4, builder).unwrap();

        hll.sparse_to_normal();

        hll.add(&0x1fffffffffffffff);
        assert_eq!(hll.registers.as_ref().unwrap().get(1), 1);

        hll.add(&0xffffffffffffffff);
        assert_eq!(hll.registers.as_ref().unwrap().get(0xf), 1);

        hll.add(&0x00ffffffffffffff);
        assert_eq!(hll.registers.as_ref().unwrap().get(0), 5);
    }

    #[test]
    fn test_sparse_encode_hash() {
        let builder = PassThroughHasherBuilder {};

        let hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(14, builder).unwrap();

        //                 < ... 14 ... > .. 25 .. >
        let index: u64 = 0b0000000000111000000000000;

        let hash: u64 = 0b1101;

        let hash_code = hll.encode_hash((index << 64 - 25) | hash);

        assert_eq!(hash_code, (index << 7) as u32 | (35 + 1 << 1) | 1);

        //                 < ... 14 ... > .. 25 .. >
        let index: u64 = 0b0000000000111000000000010;

        let hash: u64 = 0b1101;

        let hash_code = hll.encode_hash((index << 64 - 25) | hash);

        assert_eq!(hash_code, (index << 1) as u32);
    }

    #[test]
    fn test_sparse_decode_hash() {
        let builder = PassThroughHasherBuilder {};

        let hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(8, builder).unwrap();

        let (zeros, index) =
            hll.decode_hash(hll.encode_hash(0xffffff8000000000));

        assert_eq!((zeros, index), (1, 0xff));

        let (zeros, index) =
            hll.decode_hash(hll.encode_hash(0xff00000000000000));

        assert_eq!((zeros, index), (57, 0xff));

        let (zeros, index) =
            hll.decode_hash(hll.encode_hash(0xff30000000000000));

        assert_eq!((zeros, index), (3, 0xff));

        let (zeros, index) =
            hll.decode_hash(hll.encode_hash(0xaa10000000000000));

        assert_eq!((zeros, index), (4, 0xaa));

        let (zeros, index) =
            hll.decode_hash(hll.encode_hash(0xaa0f000000000000));

        assert_eq!((zeros, index), (5, 0xaa));
    }

    #[test]
    fn test_sparse_merge_sparse() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        let hashes: [u64; 3] =
            [0xf000017000000000, 0x000fff8f00000000, 0x0f00017000000000];

        let hash_codes: [u32; 3] = [
            hll.encode_hash(hashes[0]),
            hll.encode_hash(hashes[1]),
            hll.encode_hash(hashes[2]),
        ];

        // Insert a couple of hashes.
        hll.add(&hashes[0]);

        assert!(hll.tmpset.contains(&hash_codes[0]));

        hll.add(&hashes[1]);

        assert!(hll.tmpset.contains(&hash_codes[1]));

        assert_eq!(hll.tmpset.len(), 2);

        assert_eq!(hll.sparse.len(), 0);

        // Merge and check hashes.
        hll.merge_sparse();

        assert_eq!(hll.sparse.count(), 2);

        assert_eq!(hll.tmpset.len(), 0);

        let hll_hash_codes: Vec<u32> = hll.sparse.into_iter().collect();

        assert_eq!(hll_hash_codes, vec![hash_codes[1], hash_codes[0]]);

        // Insert another hash.
        hll.add(&hashes[2]);

        assert!(hll.tmpset.contains(&hash_codes[2]));

        // Merge and check hashes again.
        hll.merge_sparse();

        assert_eq!(hll.sparse.count(), 3);

        assert_eq!(hll.tmpset.len(), 0);

        let hll_hash_codes: Vec<u32> = hll.sparse.into_iter().collect();

        assert_eq!(
            hll_hash_codes,
            vec![hash_codes[1], hash_codes[2], hash_codes[0]]
        );
    }

    #[test]
    fn test_sparse_merge_to_normal() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(7, builder).unwrap();

        // We have 5 registers every 4 bytes
        // (1 << 7) * 4 / 5 = 102
        for i in 0u64..102 {
            hll.add(&(i << 39));
            hll.count();
        }

        hll.add(&1);

        assert!(hll.registers.is_none());

        hll.count();

        assert!(hll.registers.is_some());

        assert_eq!(hll.tmpset.len(), 0);

        assert_eq!(hll.sparse.len(), 0);
    }

    #[test]
    fn test_sparse_trigger_sparse_to_normal() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(4, builder).unwrap();

        // We have 5 registers every 4 bytes
        // (1 << 4) * 4 / 5 = 12
        for i in 0u64..12 {
            hll.add(&(1 << i));
        }

        assert!(hll.registers.is_none());

        hll.add(&(1 << 13));

        assert!(hll.registers.is_some());

        assert_eq!(hll.tmpset.len(), 0);

        assert_eq!(hll.sparse.len(), 0);
    }

    #[test]
    fn test_sparse_sparse_to_normal() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        hll.add(&0x00010fffffffffff);

        assert_eq!(hll.count() as u64, 1);

        hll.merge_sparse();

        hll.sparse_to_normal();

        assert_eq!(hll.count() as u64, 1);

        assert!(hll.registers.is_some());

        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        hll.add(&0x00010fffffffffff);
        hll.add(&0x0002ffffffffffff);
        hll.add(&0x0003000000000000);
        hll.add(&0x0003000000000001);
        hll.add(&0xff03700000000000);
        hll.add(&0xff03080000000000);

        hll.merge_sparse();

        hll.sparse_to_normal();

        assert_eq!(hll.count() as u64, 4);

        assert_eq!(hll.registers.as_ref().unwrap().get(1), 5);

        assert_eq!(hll.registers.as_ref().unwrap().get(2), 1);

        assert_eq!(hll.registers.as_ref().unwrap().get(3), 49);

        assert_eq!(hll.registers.as_ref().unwrap().get(0xff03), 5);
    }

    #[test]
    fn test_sparse_count() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        let hashes: [u64; 6] = [
            0x00010fffffffffff,
            0x00020fffffffffff,
            0x00030fffffffffff,
            0x00040fffffffffff,
            0x00050fffffffffff,
            0x00050fffffffffff,
        ];

        for hash in &hashes {
            hll.add(hash);
        }

        // Calls a merge_sparse().
        hll.count();

        let hash_codes: Vec<u32> = hll.sparse.into_iter().collect();

        let expected_hash_codes: Vec<u32> =
            hashes.iter().map(|hash| hll.encode_hash(*hash)).collect();

        assert_eq!(hash_codes.as_slice(), &expected_hash_codes[..5]);

        assert_eq!(hll.count() as u64, 5);
    }

    #[test]
    fn test_estimate_bias() {
        let builder = PassThroughHasherBuilder {};

        let hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(4, builder).unwrap();

        let bias = hll.estimate_bias(14.0988);

        assert!((bias - 7.5988).abs() <= 1e-5);

        let bias = hll.estimate_bias(10.0);

        assert!((bias - 10.0).abs() < 1e-5);

        let bias = hll.estimate_bias(80.0);

        assert!((bias - (-1.7606)).abs() < 1e-5);

        let builder = PassThroughHasherBuilder {};

        let hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        let bias = hll.estimate_bias(55391.4373);

        assert!((bias - 39416.9373).abs() < 1e-5);

        let builder = PassThroughHasherBuilder {};

        let hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(18, builder).unwrap();

        let bias = hll.estimate_bias(275468.768);

        assert!((bias - 118181.769).abs() <= 1e-3);

        let bias = hll.estimate_bias(587532.522);

        assert!((bias - 23922.523).abs() <= 1e-3);

        let bias = hll.estimate_bias(1205430.993);

        assert!((bias - (-434.006000000052)).abs() <= 1e-3);

        let bias = hll.estimate_bias(1251260.649);

        assert!((bias - (-479.351000000024)).abs() <= 1e-3);
    }

    #[test]
    fn test_estimate_bias_count() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(4, builder).unwrap();

        hll.sparse_to_normal();

        for i in 0u64..10 {
            hll.add(&((i << 60) + 0xfffffffffffffff));
        }

        assert!((10.0 - hll.count()).abs() < 1.0);
    }

    #[test]
    fn test_merge_error() {
        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();
        let other: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(12, PassThroughHasherBuilder {}).unwrap();

        assert_eq!(
            hll.merge(&other),
            Err(HyperLogLogError::IncompatiblePrecision)
        );
    }

    #[test]
    fn test_merge_both_sparse() {
        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();
        let mut other: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();

        other.add(&0x00010fffffffffff);
        other.add(&0x00020fffffffffff);
        other.add(&0x00030fffffffffff);
        other.add(&0x00040fffffffffff);
        other.add(&0x00050fffffffffff);
        other.add(&0x00050fffffffffff);

        assert_eq!(other.count().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(hll.is_sparse() && other.is_sparse());

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(hll.is_sparse() && other.is_sparse());

        other.add(&0x00060fffffffffff);
        other.add(&0x00070fffffffffff);
        other.add(&0x00080fffffffffff);
        other.add(&0x00090fffffffffff);
        other.add(&0x000a0fffffffffff);

        assert_eq!(other.count().trunc() as u64, 10);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 10);

        assert!(hll.is_sparse() && other.is_sparse());
    }

    #[test]
    fn test_merge_both_normal() {
        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();
        let mut other: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();

        hll.sparse_to_normal();
        other.sparse_to_normal();

        other.add(&0x00010fffffffffff);
        other.add(&0x00020fffffffffff);
        other.add(&0x00030fffffffffff);
        other.add(&0x00040fffffffffff);
        other.add(&0x00050fffffffffff);
        other.add(&0x00050fffffffffff);

        assert_eq!(other.count().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(!hll.is_sparse() && !other.is_sparse());

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(!hll.is_sparse() && !other.is_sparse());

        other.add(&0x00060fffffffffff);
        other.add(&0x00070fffffffffff);
        other.add(&0x00080fffffffffff);
        other.add(&0x00090fffffffffff);
        other.add(&0x000a0fffffffffff);

        assert_eq!(other.count().trunc() as u64, 10);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 10);

        assert!(!hll.is_sparse() && !other.is_sparse());
    }

    #[test]
    fn test_merge_sparse_to_normal() {
        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();
        let mut other: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();

        hll.sparse_to_normal();

        other.add(&0x00010fffffffffff);
        other.add(&0x00020fffffffffff);
        other.add(&0x00030fffffffffff);
        other.add(&0x00040fffffffffff);
        other.add(&0x00050fffffffffff);
        other.add(&0x00050fffffffffff);

        assert_eq!(other.count().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(!hll.is_sparse() && other.is_sparse());

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(!hll.is_sparse() && other.is_sparse());

        other.add(&0x00060fffffffffff);
        other.add(&0x00070fffffffffff);
        other.add(&0x00080fffffffffff);
        other.add(&0x00090fffffffffff);
        other.add(&0x000a0fffffffffff);

        assert_eq!(other.count().trunc() as u64, 10);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 10);

        assert!(!hll.is_sparse() && other.is_sparse());
    }

    #[test]
    fn test_merge_normal_to_sparse() {
        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();
        let mut other: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, PassThroughHasherBuilder {}).unwrap();

        other.sparse_to_normal();

        other.add(&0x00010fffffffffff);
        other.add(&0x00020fffffffffff);
        other.add(&0x00030fffffffffff);
        other.add(&0x00040fffffffffff);
        other.add(&0x00050fffffffffff);
        other.add(&0x00050fffffffffff);

        assert_eq!(other.count().trunc() as u64, 5);

        let res = hll.merge(&other);

        assert_eq!(res, Ok(()));

        assert_eq!(hll.count().trunc() as u64, 5);

        assert!(!hll.is_sparse() && !other.is_sparse());
    }

    #[test]
    fn test_serialization() {
        let builder = PassThroughHasherBuilder {};

        let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            HyperLogLogPlus::new(16, builder).unwrap();

        hll.add(&0x00010fffffffffff);
        hll.add(&0x00020fffffffffff);
        hll.add(&0x00030fffffffffff);
        hll.add(&0x00040fffffffffff);
        hll.add(&0x00050fffffffffff);
        hll.add(&0x00050fffffffffff);

        assert_eq!(hll.count().trunc() as usize, 5);

        let serialized = serde_json::to_string(&hll).unwrap();

        let mut deserialized: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.count().trunc() as usize, 5);

        deserialized.add(&0x00060fffffffffff);

        assert_eq!(deserialized.count().trunc() as usize, 6);

        hll.sparse_to_normal();

        assert_eq!(hll.count().trunc() as usize, 5);

        let serialized = serde_json::to_string(&hll).unwrap();

        let mut deserialized: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.count().trunc() as usize, 5);

        deserialized.add(&0x00060fffffffffff);

        assert_eq!(deserialized.count().trunc() as usize, 6);
    }

    #[cfg(feature = "bench-units")]
    mod benches {
        extern crate test;

        use super::*;
        use rand::prelude::*;
        use test::{black_box, Bencher};

        #[bench]
        fn bench_plus_add_normal(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
                HyperLogLogPlus::new(16, builder).unwrap();

            hll.sparse_to_normal();

            b.iter(|| {
                for i in 0u64..1000 {
                    hll.add(&(u64::max_value() - i));
                }
            })
        }

        #[bench]
        fn bench_add_normal_with_hash(b: &mut Bencher) {
            let mut rng = rand::thread_rng();

            let workload: Vec<String> = (0..2000)
                .map(|_| {
                    format!("- {} - {} -", rng.gen::<u64>(), rng.gen::<u64>())
                })
                .collect();

            b.iter(|| {
                let mut hll: HyperLogLogPlus<&String, DefaultBuildHasher> =
                    HyperLogLogPlus::new(16, DefaultBuildHasher {}).unwrap();

                hll.sparse_to_normal();

                for val in &workload {
                    hll.add(&val);
                }

                let val = hll.count();

                black_box(val);
            })
        }

        #[bench]
        fn bench_plus_count_normal(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
                HyperLogLogPlus::new(16, builder).unwrap();

            hll.sparse_to_normal();

            b.iter(|| {
                let count = hll.count();
                black_box(count);
            })
        }

        #[bench]
        fn bench_plus_merge_sparse(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let mut hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
                HyperLogLogPlus::new(16, builder).unwrap();

            for i in 0u64..500 {
                hll.add(&(i << 39));
            }

            assert_eq!(hll.tmpset.len(), 500);

            let set = hll.tmpset.clone();

            b.iter(|| {
                hll.tmpset = set.clone();
                hll.merge_sparse()
            });

            assert!(hll.registers.is_none());

            assert_eq!(hll.tmpset.len(), 0);
        }

        #[bench]
        fn bench_estimate_bias(b: &mut Bencher) {
            let builder = PassThroughHasherBuilder {};

            let hll: HyperLogLogPlus<u64, PassThroughHasherBuilder> =
                HyperLogLogPlus::new(18, builder).unwrap();

            b.iter(|| {
                let bias = hll.estimate_bias(275468.768);
                black_box(bias);
                let bias = hll.estimate_bias(587532.522);
                black_box(bias);
                let bias = hll.estimate_bias(1205430.993);
                black_box(bias);
                let bias = hll.estimate_bias(1251260.649);
                black_box(bias);
            });
        }
    }
}
