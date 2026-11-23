const SCALE: f64 = 0.797_884_560_802_865_4;
fn main() {
 let old32 = 0.797_884_560_802_865_4 as f32; let old64 = 0.797_884_560_802_865_4 as f64;
 assert_eq!(old32.to_bits(), (SCALE as f32).to_bits());
 assert_eq!(old64.to_bits(), SCALE.to_bits());
 println!("coefficient f32_bits={:#x} f64_bits={:#x}", old32.to_bits(), old64.to_bits());
}
