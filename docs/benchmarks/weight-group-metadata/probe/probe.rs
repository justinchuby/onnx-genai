#[cfg(test)]
mod allocation_probe {
    use super::*;
    use crate::group_allocation_probe::observed;
    use crate::kernels::testutil::RecordingGovernor;
    use onnx_runtime_memory_api::Tier;
    use onnx_runtime_memory_governor::{HolderId, MemoryGovernor};
    #[derive(Clone,PartialEq,Eq,Hash)]
    struct Key(usize);
    impl WeightCacheKey for Key {fn weight_bytes(&self)->Result<usize>{Ok(16)}}
    #[test]
    fn warm_group_heap_calls() {
        let positive=observed(||drop(std::hint::black_box(Box::new([3u8;1024]))));
        assert!(positive.0>0 && positive.1>0);
        println!("CONTROL allocations={} frees={}",positive.0,positive.1);
        for len in [0usize,1,2,3,4,8] {
            let keys:Vec<_>=(0..len).map(Key).collect();
            let specs:Vec<_>=keys.iter().map(|key|WeightSpec::cached(key).unwrap()).collect();
            let books=RecordingGovernor::new_host((len*16) as u64);
            let cache=WeightCache::<Key>::new(books.weight_memory(HolderId::new(13)),len*16);
            let WeightLoad::Ready(cold)=cache.load_group(&specs,|index,out|{out.fill(index as f32+1.);Ok(())}).unwrap() else {panic!("cold group must fit")};
            assert_eq!(books.used(Tier::Host),(len*16) as u64);
            let before=(books.claims(),books.releases());
            let counts=observed(||{
                for _ in 0..2048 {
                    let WeightLoad::Ready(warm)=cache.load_group(&specs,|_,_|panic!("warm hit must not build")).unwrap() else {panic!("warm group must hit")};
                    assert_eq!(warm.weights.len(),len);
                    for (index,(a,b)) in cold.weights.iter().zip(&warm.weights).enumerate(){
                        assert!(Arc::ptr_eq(a,b));assert_eq!(b.as_slice(),&[index as f32+1.;4]);
                    }
                    assert!(warm.statuses.iter().all(|s|*s==WeightCacheStatus::Hit));
                    std::hint::black_box(&warm);
                }
            });
            assert_eq!((books.claims(),books.releases()),before);
            assert_eq!(cache.stats(),(len,len*16));
            drop(cache);assert_eq!(books.used(Tier::Host),(len*16) as u64);
            drop(cold);assert_eq!(books.used(Tier::Host),0);
            println!("GROUP members={len} iterations=2048 allocations={} frees={} issuer_delta=0 final_used=0",counts.0,counts.1);
        }
    }
}
