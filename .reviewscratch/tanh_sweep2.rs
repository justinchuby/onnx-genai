mod tanh_c {
    pub const ALPHA_13: f32 = -2.76076847742355e-16;
    pub const ALPHA_11: f32 = 2.00018790482477e-13;
    pub const ALPHA_9: f32 = -8.60467152213735e-11;
    pub const ALPHA_7: f32 = 5.12229709037114e-08;
    pub const ALPHA_5: f32 = 1.48572235717979e-05;
    pub const ALPHA_3: f32 = 6.37261928875436e-04;
    pub const ALPHA_1: f32 = 4.89352455891786e-03;
    pub const BETA_6: f32 = 1.19825839466702e-06;
    pub const BETA_4: f32 = 1.18534705686654e-04;
    pub const BETA_2: f32 = 2.26843463243900e-03;
    pub const BETA_0: f32 = 4.89352518554385e-03;
}

// Faithful FMA version (matches _mm256_fmadd_ps).
fn rational_fma(x: f32) -> f32 {
    use tanh_c::*;
    let v = x.clamp(-9.0, 9.0);
    let v2 = v * v;
    let mut p = v2.mul_add(ALPHA_13, ALPHA_11);
    p = p.mul_add(v2, ALPHA_9);
    p = p.mul_add(v2, ALPHA_7);
    p = p.mul_add(v2, ALPHA_5);
    p = p.mul_add(v2, ALPHA_3);
    p = p.mul_add(v2, ALPHA_1);
    p = p * v;
    let mut q = v2.mul_add(BETA_6, BETA_4);
    q = q.mul_add(v2, BETA_2);
    q = q.mul_add(v2, BETA_0);
    p / q
}

// Non-fused version: each a*b+c rounds twice.
fn rational_nofma(x: f32) -> f32 {
    use tanh_c::*;
    let v = x.clamp(-9.0, 9.0);
    let v2 = v * v;
    let mut p = v2 * ALPHA_13 + ALPHA_11;
    p = p * v2 + ALPHA_9;
    p = p * v2 + ALPHA_7;
    p = p * v2 + ALPHA_5;
    p = p * v2 + ALPHA_3;
    p = p * v2 + ALPHA_1;
    p = p * v;
    let mut q = v2 * BETA_6 + BETA_4;
    q = q * v2 + BETA_2;
    q = q * v2 + BETA_0;
    p / q
}

fn sweep(name: &str, f: fn(f32) -> f32) {
    let mut count = 0u64;
    let mut first: Option<f32> = None;
    let mut last: Option<f32> = None;
    let mut peak = f32::NEG_INFINITY;
    let mut peak_x_first = 0.0f32;
    let mut peak_x_last = 0.0f32;
    let mut b = 8.0f32.to_bits();
    let end = 9.0f32.to_bits();
    while b <= end {
        let x = f32::from_bits(b);
        let r = f(x);
        if r > 1.0 {
            count += 1;
            if first.is_none() { first = Some(x); }
            last = Some(x);
        }
        if r > peak { peak = r; peak_x_first = x; peak_x_last = x; }
        else if r == peak { peak_x_last = x; }
        b += 1;
    }
    println!("== {name} ==");
    println!("  count > 1.0 = {count}");
    println!("  span        = [{:?}, {:?}]", first, last);
    println!("  peak        = {:?} (0x{:08X})", peak, peak.to_bits());
    println!("  peak x range= [{:?} (0x{:08X}), {:?} (0x{:08X})]",
        peak_x_first, peak_x_first.to_bits(), peak_x_last, peak_x_last.to_bits());
    println!("  f(8.442762) = {:?} (0x{:08X})", f(8.442762), f(8.442762).to_bits());
    println!("  f(8.052297) = {:?} (0x{:08X})", f(8.052297), f(8.052297).to_bits());
    println!("  f(9.0)      = {:?} (0x{:08X})", f(9.0), f(9.0).to_bits());
}

fn main() {
    sweep("FMA (matches this module)", rational_fma);
    sweep("no-FMA (double rounding)", rational_nofma);
}
