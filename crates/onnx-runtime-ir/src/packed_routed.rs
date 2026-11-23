//! Attribute contracts for the standard opset 29 packing and routing operators.

use crate::{Attribute, DataType, Node};

pub const OPS: [&str; 3] = ["Pack", "Unpack", "GroupedMatMul"];
pub const WORD_DTYPES: &[DataType] = &[DataType::Uint8, DataType::Uint32];

#[derive(Clone, Copy, Debug)]
pub struct Packing {
    axis: i64,
    pub radix: u64,
    pub group: usize,
    pub limit: u64,
    pub bits: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Pack(Packing, DataType),
    Unpack(Packing),
    GroupedMatMul { transposed: bool },
}

impl Op {
    pub fn from_node(node: &Node) -> Result<Self, String> {
        let name = node.op_type.as_str();
        let int = |key| match node.attr(key) {
            None => Ok(None),
            Some(Attribute::Int(n)) => Ok(Some(*n)),
            _ => Err(format!("{name}: {key} must be an INT attribute")),
        };
        if name == "GroupedMatMul" {
            return Ok(Self::GroupedMatMul {
                transposed: int("transB")?.unwrap_or(0) != 0,
            });
        }
        let (radix, group) = match (int("bits")?, int("radix")?, int("group_size")?) {
            (Some(bits @ 1..=8), None, None) => (1u64 << bits, 1),
            (None, Some(radix @ 2..=256), Some(group @ 1..=62)) => (radix as u64, group),
            _ => {
                return Err(format!(
                    "{name}: specify either bits in [1, 8], or radix in [2, 256] and positive group_size with radix^group_size <= INT64_MAX"
                ));
            }
        };
        let limit = radix
            .checked_pow(group as u32)
            .filter(|&n| n <= i64::MAX as u64)
            .ok_or_else(|| format!("{name}: radix {radix} raised to group_size {group} exceeds INT64_MAX; reduce radix or group_size"))?;
        let packing = Packing {
            axis: int("axis")?.unwrap_or(-1),
            radix,
            group: group as usize,
            limit,
            bits: (64 - (limit - 1).leading_zeros()) as usize,
        };
        match name {
            "Pack" => match int("output_type")?.unwrap_or(2) {
                2 => Ok(Self::Pack(packing, DataType::Uint8)),
                12 => Ok(Self::Pack(packing, DataType::Uint32)),
                n => Err(format!(
                    "Pack: output_type {n} is invalid; use UINT8 (2) or UINT32 (12)"
                )),
            },
            "Unpack" => Ok(Self::Unpack(packing)),
            _ => Err(format!("{name}: expected Pack, Unpack or GroupedMatMul")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pack(..) => "Pack",
            Self::Unpack(..) => "Unpack",
            Self::GroupedMatMul { .. } => "GroupedMatMul",
        }
    }

    pub fn validate_dtypes(self, dtypes: &[DataType]) -> Result<(), String> {
        let get = |i| dtypes.get(i).copied().unwrap_or(DataType::Undefined);
        let dir = DataType::Undefined;
        let x = get(0);
        let fail = |i, valid: &str| {
            Err(format!(
                "{}: input {i} has dtype {:?}; expected {valid}",
                self.name(),
                get(i)
            ))
        };
        match self {
            Self::Pack(..) if x != dir && x != DataType::Uint8 => fail(0, "UINT8 codes"),
            Self::Unpack(..) if x != dir && !WORD_DTYPES.contains(&x) => {
                fail(0, "UINT8 or UINT32 words")
            }
            Self::Unpack(..) if ![dir, DataType::Int64].contains(&get(1)) => {
                fail(1, "an INT64 scalar count")
            }
            Self::GroupedMatMul { .. } => {
                let mut known = None;
                for i in [0, 1, 3] {
                    let dtype = get(i);
                    if dtype == dir {
                        continue;
                    }
                    if !crate::standard_math::DTYPES.contains(&dtype)
                        || known.is_some_and(|first| first != dtype)
                    {
                        return fail(i, "matching FLOAT16, BFLOAT16, FLOAT or DOUBLE operands");
                    }
                    known = Some(dtype);
                }
                if ![dir, DataType::Int32, DataType::Int64].contains(&get(2)) {
                    return fail(2, "INT32 or INT64 group indices");
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Packing {
    pub fn axis(self, rank: usize) -> Result<usize, String> {
        let r = rank as i64;
        if !(-r..r).contains(&self.axis) {
            return Err(format!(
                "axis {} is invalid for rank {rank}; use a rank >= 1 and axis in [-rank, rank)",
                self.axis
            ));
        }
        Ok(if self.axis < 0 {
            r + self.axis
        } else {
            self.axis
        } as usize)
    }

    pub fn words(self, count: usize, word_bits: usize) -> Result<usize, String> {
        count
            .div_ceil(self.group)
            .checked_mul(self.bits)
            .map(|bits| bits.div_ceil(word_bits))
            .ok_or_else(|| format!("count {count} is too large for this packed layout"))
    }
}
