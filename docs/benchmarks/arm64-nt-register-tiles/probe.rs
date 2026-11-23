//! Serial native NT entry and optional same-binary MLAS reference.
use onnx_runtime_ep_cpu::{backend::CpuBackend, kernels::matmul::gemm_bt_serial_with_backend};
use std::{hint::black_box, time::Instant};
fn usage() -> (f64,u64) {
 let mut u=std::mem::MaybeUninit::<libc::rusage>::uninit();assert_eq!(unsafe{libc::getrusage(libc::RUSAGE_SELF,u.as_mut_ptr())},0);let u=unsafe{u.assume_init()};
 (u.ru_utime.tv_sec as f64+u.ru_utime.tv_usec as f64/1e6+u.ru_stime.tv_sec as f64+u.ru_stime.tv_usec as f64/1e6,if cfg!(target_os="macos"){u.ru_maxrss as u64}else{u.ru_maxrss as u64*1024})
}
#[inline(never)]
fn control(input:&[f32],output:&mut[f32]) {for (a,b) in input.iter().zip(output){*b=a.max(0.0);}}
fn run_backend(backend:CpuBackend,a:&[f32],b:&[f32],c:&mut[f32],m:usize,k:usize,n:usize,reps:usize)->(f64,f64){
 let start=usage().0;let wall=Instant::now();for _ in 0..reps{gemm_bt_serial_with_backend(black_box(a),black_box(b),black_box(&mut *c),m,k,n,backend).unwrap();}black_box(c);((usage().0-start)*1000.,wall.elapsed().as_secs_f64()*1000.)
}
fn main(){
 let args=std::env::args().skip(1).collect::<Vec<_>>();assert_eq!(args.len(),6,"generic|paired M K N reps output-prefix");let paired=args[0]=="paired";assert!(paired||args[0]=="generic");if paired{assert!(onnx_runtime_ep_cpu::backend_ab::mlas_available(),"paired measurement requires the explicit MLAS research build");}
 let nums=args[1..5].iter().map(|x|x.parse::<usize>().unwrap()).collect::<Vec<_>>();let(m,k,n,reps)=(nums[0],nums[1],nums[2],nums[3]);assert!(m>0&&k>0&&n>0&&reps>0);
 let mut state=0x928b_3465u32;let mut next=||{state=state.wrapping_mul(1664525).wrapping_add(1013904223);(state>>16)as i16 as f32/262144.};
 let a:Vec<_>=(0..m*k).map(|_|next()).collect();let b:Vec<_>=(0..n*k).map(|_|next()).collect();let mut c=vec![0f32;m*n];#[allow(unused_mut)] let mut reference=vec![0f32;m*n];
 let oracle:Vec<f64>=(0..m*n).map(|i|{let r=i/n;let col=i%n;(0..k).map(|p|a[r*k+p]as f64*b[col*k+p]as f64).sum()}).collect();
 gemm_bt_serial_with_backend(&a,&b,&mut c,m,k,n,CpuBackend::Generic).unwrap();
 #[cfg(feature="mlas")]if paired{gemm_bt_serial_with_backend(&a,&b,&mut reference,m,k,n,CpuBackend::Mlas).unwrap();}
 let error=|values:&[f32]|{values.iter().zip(&oracle).map(|(&x,&y)|{let d=(x as f64-y).abs();assert!(x.is_finite()&&d<=1e-4+1e-5*y.abs(),"f64 oracle deviation {x} vs {y}");d}).fold(0.0,f64::max)};
 let max_abs=error(&c);let ref_abs=if paired{error(&reference)}else{0.0};
 let warm=usage().0;while usage().0-warm<0.1{run_backend(CpuBackend::Generic,&a,&b,&mut c,m,k,n,1);#[cfg(feature="mlas")]if paired{run_backend(CpuBackend::Mlas,&a,&b,&mut reference,m,k,n,1);}}
 let mut cpu=Vec::new();let mut wall=Vec::new();#[allow(unused_mut)] let mut ref_cpu=Vec::<f64>::new();#[allow(unused_mut)] let mut ref_wall=Vec::<f64>::new();
 for index in 0..9 {
  #[cfg(feature="mlas")]if paired&&index%2==1{let(t,w)=run_backend(CpuBackend::Mlas,&a,&b,&mut reference,m,k,n,reps);ref_cpu.push(t);ref_wall.push(w);}
  let(t,w)=run_backend(CpuBackend::Generic,&a,&b,&mut c,m,k,n,reps);cpu.push(t);wall.push(w);
  #[cfg(feature="mlas")]if paired&&index%2==0{let(t,w)=run_backend(CpuBackend::Mlas,&a,&b,&mut reference,m,k,n,reps);ref_cpu.push(t);ref_wall.push(w);}
  #[cfg(not(feature="mlas"))]let _=index;
 }
 let peak=usage().1;let data=|v:&[f32]|v.iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<_>>();std::fs::write(format!("{}.native.f32",args[5]),data(&c)).unwrap();if paired{std::fs::write(format!("{}.reference.f32",args[5]),data(&reference)).unwrap();}
 let argmax:Vec<_>=c.chunks_exact(n).map(|row|(0..n).max_by(|&x,&y|row[x].total_cmp(&row[y]).then_with(||y.cmp(&x))).unwrap()).collect();
 let control_input:Vec<_>=(0..16384).map(|i|(i%31)as f32-15.).collect();let mut control_output=vec![0f32;16384];for _ in 0..1000{control(&control_input,&mut control_output);black_box(&control_output);}
 let mut controls=Vec::new();for _ in 0..9{let start=usage().0;for _ in 0..100000{control(black_box(&control_input),black_box(&mut control_output));black_box(&control_output);}controls.push((usage().0-start)*1000.);}
 println!("{{\"m\":{m},\"k\":{k},\"n\":{n},\"reps\":{reps},\"paired\":{paired},\"cpu_batch_ms\":{cpu:?},\"wall_batch_ms\":{wall:?},\"reference_cpu_batch_ms\":{ref_cpu:?},\"reference_wall_batch_ms\":{ref_wall:?},\"control_batch_ms\":{controls:?},\"max_abs_f64_error\":{max_abs},\"reference_max_abs_f64_error\":{ref_abs},\"absolute_peak_rss_bytes\":{peak},\"argmax\":{argmax:?}}}");
}
