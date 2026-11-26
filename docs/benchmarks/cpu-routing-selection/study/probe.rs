#[cfg(test)]
mod selection_comparison_probe {
    use super::*;
    #[test]
    fn count_ordering_work_without_assuming_latency() {
        let mut scratch = RoutingScratch::default();
        for experts in [0usize, 1, 2, 8, 64, 256, 1024] {
            for pattern in ["random", "descending", "ties", "nonfinite"] {
                let mut seed=0xa713_481bu32;
                let logits:Vec<_>=(0..experts).map(|i| match pattern {
                    "random"=>{seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);f32::from_bits(0x3f80_0000|(seed&0x007f_ffff))-1.},
                    "descending"=>(experts-i) as f32,
                    "ties"=>1.,
                    _=>[f32::from_bits(0x7fc0_0032),f32::from_bits(0xffc0_0021),f32::INFINITY,f32::NEG_INFINITY,0.,-0.,f32::MAX,-f32::MAX][i%8],
                }).collect();
                let mut ks=vec![0,1,2,experts/2,experts.saturating_sub(1),experts,experts+3];ks.sort_unstable();ks.dedup();
                for k in ks {
                    ROUTE_ORDER_COMPARISONS.with(|c|c.set(0));
                    let baseline=scratch.fullsort_weights(&logits,None,k,true).to_vec();
                    let before=ROUTE_ORDER_COMPARISONS.with(std::cell::Cell::get);
                    ROUTE_ORDER_COMPARISONS.with(|c|c.set(0));
                    let candidate=scratch.weights(&logits,None,k,true).to_vec();
                    let after=ROUTE_ORDER_COMPARISONS.with(std::cell::Cell::get);
                    let expected=reference_routing_weights(&logits,None,k,true);
                    assert_eq!(candidate.len(),k.min(experts));
                    for ((b,a),r) in baseline.iter().zip(&candidate).zip(&expected){
                        assert_eq!(b.0,a.0);assert_eq!(a.0,r.0);
                        assert_eq!(b.1.to_bits(),a.1.to_bits());assert_eq!(a.1.to_bits(),r.1.to_bits());
                    }
                    let pairs:Vec<_>=candidate.iter().map(|&(i,w)|(i,w.to_bits())).collect();
                    println!("CASE experts={experts} k={k} pattern={pattern} baseline_comparisons={before} candidate_comparisons={after}");
                    println!("OUTPUT experts={experts} k={k} pattern={pattern} pairs={pairs:?}");
                }
            }
        }
    }
}
