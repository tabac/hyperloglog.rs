use serde::{Deserialize, Serialize};

// A Vector of bytes containing variable length encoded unsigned integers.
#[derive(Serialize, Deserialize, Debug)]
pub struct VarIntVec(Vec<u8>);

// A Vector containing difference encoded unsigned integers
// stored in a `VarIntVec`.
//
// Numbers stored are assumed to be in increasing order, hence the
// difference between a new number and `last` will always be positive.
#[derive(Serialize, Deserialize, Debug)]
pub struct DifIntVec {
    // The count of numbers stored.
    count: usize,
    // The last number inserted.
    last:  u32,
    // The inner Varint encoded vector.
    buf:   VarIntVec,
}

pub struct VarIntVecIntoIter<'a> {
    index: usize,
    inner: &'a VarIntVec,
}

pub struct DifIntVecIntoIter<'a> {
    index: usize,
    last:  u32,
    inner: &'a DifIntVec,
}

impl VarIntVec {
    const SEVEN_LSB_MASK: u32 = (1 << 7) - 1;

    const VAR_INT_MASK: u32 = !Self::SEVEN_LSB_MASK;

    const MSB_MASK: u8 = 1 << 7;

    pub fn new() -> Self {
        VarIntVec(Vec::new())
    }

    pub fn with_capacity(cap: usize) -> Self {
        VarIntVec(Vec::with_capacity(cap))
    }

    #[inline] // Varint encodes `val` and pushes it into the vector.
    pub fn push(&mut self, mut val: u32) {
        while val & Self::VAR_INT_MASK != 0 {
            self.0
                .push((val & Self::SEVEN_LSB_MASK) as u8 | Self::MSB_MASK);

            val >>= 7;
        }

        self.0.push((val & Self::SEVEN_LSB_MASK) as u8);
    }

    #[inline] // Varint decodes a number starting at `index`.
              //
              // Returns the decoded number and the index of the next
              // number in the vector.
    fn decode(&self, index: usize) -> (u32, usize) {
        let (mut i, mut val) = (0, 0);

        while self.0[index + i] & Self::MSB_MASK != 0 {
            val |=
                ((self.0[index + i] as u32) & Self::SEVEN_LSB_MASK) << (i * 7);

            i += 1;
        }

        val |= (self.0[index + i] as u32) << (i * 7);

        (val, index + i + 1)
    }
}

impl DifIntVec {
    pub fn new() -> Self {
        DifIntVec {
            count: 0,
            last:  0,
            buf:   VarIntVec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        DifIntVec {
            count: 0,
            last:  0,
            buf:   VarIntVec::with_capacity(cap),
        }
    }

    #[inline] // Difference encodes `val` and pushes it into a `VarIntVec`.
    pub fn push(&mut self, val: u32) {
        self.buf.push(val - self.last);
        self.last = val;
        self.count += 1;
    }

    #[inline] // Returns the count of numbers in the vector.
    pub fn count(&self) -> usize {
        self.count
    }

    #[inline] // Returns the inner vector's length.
    pub fn len(&self) -> usize {
        self.buf.0.len()
    }

    // Clear the inner vector.
    pub fn clear(&mut self) {
        self.count = 0;
        self.last = 0;
        self.buf.0.clear();
    }
}

impl<'a> IntoIterator for &'a VarIntVec {
    type Item = u32;
    type IntoIter = VarIntVecIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        VarIntVecIntoIter {
            index: 0,
            inner: self,
        }
    }
}

impl<'a> IntoIterator for &'a DifIntVec {
    type Item = u32;
    type IntoIter = DifIntVecIntoIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        DifIntVecIntoIter {
            index: 0,
            last:  0,
            inner: self,
        }
    }
}

impl<'a> Iterator for VarIntVecIntoIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.inner.0.len() {
            None
        } else {
            let (val, index) = self.inner.decode(self.index);

            self.index = index;

            Some(val)
        }
    }
}

impl<'a> Iterator for DifIntVecIntoIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.inner.buf.0.len() {
            None
        } else {
            let (dif, index) = self.inner.buf.decode(self.index);

            self.index = index;

            self.last += dif;

            Some(self.last)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encode() {
        let mut nums = VarIntVec::with_capacity(2);

        nums.push(7);
        nums.push(128);
        nums.push(300);

        assert_eq!(nums.0, vec![7, 128, 1, 172, 2]);

        nums.push(127);

        assert_eq!(nums.0, vec![7, 128, 1, 172, 2, 127]);

        nums.push(255);

        assert_eq!(nums.0, vec![7, 128, 1, 172, 2, 127, 255, 1]);

        nums.push(0xffffffff);

        assert_eq!(
            nums.0,
            vec![7, 128, 1, 172, 2, 127, 255, 1, 255, 255, 255, 255, 15]
        );
    }

    #[test]
    fn test_varint_decode() {
        let input: Vec<u32> = (1..256).chain(16400..16500).collect();

        let mut nums = VarIntVec::with_capacity(100);

        for num in &input {
            nums.push(*num);
        }

        let output: Vec<u32> = nums.into_iter().collect();

        assert_eq!(input, output);
    }

    #[test]
    fn test_difvarint_encode() {
        let mut nums = DifIntVec::with_capacity(2);

        // push: 7
        nums.push(7);
        // push: 128 - 7 = 121
        nums.push(128);
        // push: 300 - 128 = 172
        nums.push(300);

        assert_eq!(nums.buf.0, vec![7, 121, 172, 1]);

        // push: 555 - 300 = 255
        nums.push(555);

        assert_eq!(nums.buf.0, vec![7, 121, 172, 1, 255, 1]);

        // push: 556 - 555 = 1
        nums.push(556);

        assert_eq!(nums.buf.0, vec![7, 121, 172, 1, 255, 1, 1]);
    }

    #[test]
    fn test_difvarint_decode() {
        let input: Vec<u32> = (1..256).chain(16400..16500).collect();

        let mut nums = DifIntVec::with_capacity(100);

        for num in &input {
            nums.push(*num);
        }

        let output: Vec<u32> = nums.into_iter().collect();

        assert_eq!(input, output);
    }

    #[cfg(feature = "bench-units")]
    mod benches {
        extern crate test;

        use super::*;
        use test::Bencher;

        #[bench]
        fn bench_varint_push_min(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(1000);

            b.iter(|| buf.push(u32::min_value()));
        }

        #[bench]
        fn bench_varint_push_max(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(1000);

            b.iter(|| buf.push(u32::max_value()));
        }

        #[bench]
        fn bench_varint_decode_min(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(10);

            for _ in 0..10 {
                buf.push(u32::min_value());
            }

            b.iter(|| buf.decode(0));
        }

        #[bench]
        fn bench_varint_decode_max(b: &mut Bencher) {
            let mut buf = VarIntVec::with_capacity(10);

            for _ in 0..10 {
                buf.push(u32::max_value());
            }

            b.iter(|| buf.decode(0));
        }
    }
}
