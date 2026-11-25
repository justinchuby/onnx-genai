#[cfg(test)]
mod route_count_allocation_probe {
    use super::*;
    use crate::group_allocation_probe::observed;
    fn compare(a: &WeightOffloadMetrics, b: &WeightOffloadMetrics, active: usize, unique: usize) {
        let mut actual=a.snapshot();let expected=b.snapshot();
        actual.linux_process=expected.linux_process;
        assert_eq!(actual,expected);
        assert_eq!(actual.layer_executions,2049);
        assert_eq!(actual.active_experts,2049*active as u64);
        assert_eq!(actual.routed_tokens,2049*active as u64);
        assert_eq!(actual.unique_experts_per_batch,2049*unique as u64);
    }
    #[test]
    fn route_statistics_borrow_existing_groups_without_temporary_counts() {
        let positive=observed(||drop(std::hint::black_box(Box::new([3u8;1024]))));
        assert!(positive.0>0 && positive.1>0);
        println!("CONTROL allocations={} frees={}",positive.0,positive.1);
        for experts in [1usize,7,65] {
            for rows in [1usize,8,64] {
                let mut ks=vec![1,experts.min(3)];ks.dedup();
                for k in ks {
                    let events:Vec<_>=(0..rows).flat_map(|r|(0..k).map(move |rank|(r+k-1-rank)%experts)).collect();
                    let mut tasks=BTreeMap::<usize,Vec<(usize,usize,f32)>>::new();
                    let mut prime=BTreeMap::new();
                    for (slot,&e) in events.iter().enumerate(){
                        tasks.entry(e).or_default().push((slot/k,slot,1.));
                        *prime.entry(e).or_insert(0usize)+=1;
                    }
                    let baseline=WeightOffloadMetrics::default();
                    let candidate=WeightOffloadMetrics::default();
                    baseline.reference_record_routes(77,&prime);
                    candidate.record_routes(77,&tasks);
                    let old=observed(||{
                        for _ in 0..2048 {
                            let mut counts=BTreeMap::new();
                            for &e in &events{*counts.entry(e).or_insert(0usize)+=1;}
                            baseline.reference_record_routes(77,&counts);
                            std::hint::black_box(&counts);
                        }
                    });
                    let new=observed(||{
                        for _ in 0..2048 {
                            candidate.record_routes(77,&tasks);
                        }
                    });
                    assert!(old.0>0 && old.0==old.1);assert_eq!(new,(0,0));
                    compare(&baseline,&candidate,events.len(),tasks.len());
                    println!("CASE experts={experts} rows={rows} k={k} active={} unique={} calls=2048 baseline_alloc={} baseline_free={} candidate_alloc={} candidate_free={}",events.len(),tasks.len(),old.0,old.1,new.0,new.1);
                }
            }
        }
    }
}
