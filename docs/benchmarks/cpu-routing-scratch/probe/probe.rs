#[cfg(test)]
mod routing_allocation_probe {
    use super::{RoutingScratch, reference_routing_weights};
    use crate::group_allocation_probe::observed;
    fn check(a: &[(usize,f32)], b: &[(usize,f32)]) {
        assert_eq!(a.len(),b.len());
        for ((ai,aw),(bi,bw)) in a.iter().zip(b){assert_eq!(ai,bi);assert_eq!(aw.to_bits(),bw.to_bits());}
    }
    #[test]
    fn repeated_rows_reuse_routing_metadata() {
        let positive=observed(||drop(std::hint::black_box(Box::new([3u8;1024]))));
        assert!(positive.0>0 && positive.1>0);
        println!("CONTROL allocations={} frees={}",positive.0,positive.1);
        let mut scratch=RoutingScratch::default();
        for experts in [1usize,7,65,257] {
            let logits:Vec<_>=(0..experts).map(|i|((i*13)%17) as f32*0.125-1.).collect();
            let aggregation:Vec<_>=(0..experts).map(|i|[1.,-1.,0.,0.25][i%4]).collect();
            let mut ks=vec![1,experts];ks.dedup();
            for k in ks {
                for normalize in [false,true] {
                    for aggregated in [false,true] {
                        let weights=aggregated.then_some(aggregation.as_slice());
                        let expected=reference_routing_weights(&logits,weights,k,normalize);
                        check(scratch.weights(&logits,weights,k,normalize),&expected);
                        let before=observed(||{
                            for _ in 0..2048 {
                                let mut route=reference_routing_weights(&logits,weights,k,normalize);
                                check(&route,&expected);
                                route.sort_unstable_by_key(|&(i,_)|i);
                                std::hint::black_box(&route);
                            }
                        });
                        let after=observed(||{
                            for _ in 0..2048 {
                                let route=scratch.weights(&logits,weights,k,normalize);
                                check(route,&expected);
                                route.sort_unstable_by_key(|&(i,_)|i);
                                std::hint::black_box(route);
                            }
                        });
                        assert!(before.0>0 && before.0==before.1);
                        assert_eq!(after,(0,0));
                        println!("CASE experts={experts} k={k} normalize={normalize} aggregation={aggregated} calls=2048 baseline_alloc={} baseline_free={} candidate_alloc={} candidate_free={}",before.0,before.1,after.0,after.1);
                    }
                }
            }
        }
    }
}
