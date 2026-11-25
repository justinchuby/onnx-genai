use std::hint::black_box;
#[inline(never)]
fn value(v:f32)->f32 {v+0.0}
fn sigmoid(v:f32)->f32 { if v>=0. {1./(1.+(-v).exp())}else{let x=v.exp();x/(1.+x)}}
fn gate(g:f32)->f32 {let g=g.min(7.);g*sigmoid(g)}
fn main(){for g in [black_box(-0.0),black_box(0.0)] {let rounded=value(g); println!("{:x} {:x} {:x} {:x}",g.to_bits(),rounded.to_bits(),gate(g+0.).to_bits(),gate(rounded).to_bits());}}
