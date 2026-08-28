#![cfg(feature = "native-cuda")]

use onnx_runtime_ep_api::{
    ExternalMmapRegion, LazyWeight, LazyWeightBoundary, MmapRegionSource, ResidentWeight,
    WeightHandleError,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::DataType;

const BANK_BYTES: usize = 4096;

struct TinyQmoeMmap {
    bytes: Vec<u8>,
}

impl MmapRegionSource for TinyQmoeMmap {
    fn region_bytes(&self, region: &ExternalMmapRegion) -> Result<&[u8], WeightHandleError> {
        if region.mapping_id != 1759 {
            return Err(WeightHandleError::DeviceBinding(
                "unexpected tiny-QMoE mapping".to_string(),
            ));
        }
        let end = region
            .offset
            .checked_add(region.len)
            .ok_or_else(|| WeightHandleError::DeviceBinding("region overflow".to_string()))?;
        self.bytes
            .get(region.offset..end)
            .ok_or_else(|| WeightHandleError::DeviceBinding("region out of bounds".to_string()))
    }
}

fn tiny_qmoe_banks() -> (TinyQmoeMmap, Vec<LazyWeight>) {
    let banks = (0..2)
        .map(|bank| {
            (0..BANK_BYTES)
                .map(|index| ((bank * 61 + index * 17) & 0xff) as u8)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut bytes = Vec::with_capacity(BANK_BYTES * banks.len());
    let mut weights = Vec::with_capacity(banks.len());
    for bank in banks {
        let offset = bytes.len();
        bytes.extend_from_slice(&bank);
        let region = ExternalMmapRegion {
            mapping_id: 1759,
            offset,
            len: bank.len(),
        };
        let materialized = bank;
        weights.push(
            LazyWeight::new(
                LazyWeightBoundary::QMoe,
                DataType::Uint8,
                vec![2, BANK_BYTES / 2],
                vec![region],
                move || {
                    ResidentWeight::new(
                        DataType::Uint8,
                        vec![2, BANK_BYTES / 2],
                        materialized.clone(),
                    )
                },
            )
            .expect("valid tiny-QMoE bank"),
        );
    }
    (TinyQmoeMmap { bytes }, weights)
}

#[test]
fn tiny_qmoe_records_a_real_page_in_and_a_resident_hit() -> anyhow::Result<()> {
    let ep = match onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        Ok(ep) => ep,
        Err(error) => {
            eprintln!("skipping tiny-QMoE byte-counter fixture: CUDA unavailable: {error}");
            return Ok(());
        }
    };

    // Current main does not yet route production QMoE expert banks through the
    // residency caller. Exercise the existing QMoE-boundary cache API directly
    // instead of pretending engine counters observed traffic they did not: a
    // one-bank budget forces bank 0 miss, bank 0 hit, then bank 1 miss + eviction.
    onnx_runtime_ep_cuda::reset_global_offload_stats();
    let (mmap, banks) = tiny_qmoe_banks();
    let residency = ep.weight_residency(BANK_BYTES as u64);
    let first = residency.resident(0, &banks[0], &mmap)?;
    let mut first_bytes = vec![0u8; BANK_BYTES];
    // SAFETY: `first` owns exactly BANK_BYTES readable device bytes.
    unsafe {
        ep.runtime()
            .dtoh(&mut first_bytes, cuptr(first.device_ptr()))?
    };
    assert_eq!(first_bytes, mmap.bytes[..BANK_BYTES]);

    let hit = residency.resident(0, &banks[0], &mmap)?;
    assert_eq!(
        hit.device_ptr(),
        first.device_ptr(),
        "a hit must reuse the resident bank's device binding"
    );
    drop(first);
    drop(hit);

    let second = residency.resident(1, &banks[1], &mmap)?;
    let mut second_bytes = vec![0u8; BANK_BYTES];
    // SAFETY: `second` owns exactly BANK_BYTES readable device bytes.
    unsafe {
        ep.runtime()
            .dtoh(&mut second_bytes, cuptr(second.device_ptr()))?
    };
    assert_eq!(second_bytes, mmap.bytes[BANK_BYTES..]);
    let offload = onnx_runtime_ep_cuda::global_offload_stats();

    assert_eq!(offload.page_ins, 2, "two cold bank accesses: {offload:?}");
    assert_eq!(offload.hits, 1, "one repeated bank access: {offload:?}");
    assert_eq!(offload.htod_bytes, (BANK_BYTES * 2) as u64);
    assert_eq!(offload.hit_bytes, BANK_BYTES as u64);
    assert_eq!(
        offload.evictions, 1,
        "one-bank budget must evict: {offload:?}"
    );
    let byte_hit_rate = offload
        .zero_copy_byte_hit_rate()
        .expect("the fixture requested weight bytes");
    assert_eq!(
        byte_hit_rate,
        1.0 / 3.0,
        "one equal-sized hit plus two misses must produce an exact 1/3 byte-hit rate"
    );
    Ok(())
}
