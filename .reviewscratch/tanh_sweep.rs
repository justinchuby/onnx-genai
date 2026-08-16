// Reproduce the MLAS tanh rational (p/q) exactly as tanh_ps evaluates it,
// using f32::mul_add to match _mm256_fmadd_ps, and sweep [8,9].
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

#[inline(never)]
fn rational(x: f32) -> f32 {
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

fn main() {
    let mut count = 0u64;
    let mut first: Option<f32> = None;
    let mut last: Option<f32> = None;
    let mut peak = f32::NEG_INFINITY;
    let mut peak_x = 0.0f32;
    let mut b = 8.0f32.to_bits();
    let end = 9.0f32.to_bits();
    let mut total = 0u64;
    while b <= end {
        let x = f32::from_bits(b);
        let r = rational(x);
        total += 1;
        if r > 1.0 {
            count += 1;
            if first.is_none() {
                first = Some(x);
            }
            last = Some(x);
            if r > peak {
                peak = r;
                peak_x = x;
            }
        }
        b += 1;
    }
    println!("total f32 in [8,9]      = {total}");
    println!("count with p/q > 1.0    = {count}");
    println!("span first              = {:?} ({})", first, first.map(|v| format!("0x{:08X}", v.to_bits())).unwrap_or_default());
    println!("span last               = {:?} ({})", last, last.map(|v| format!("0x{:08X}", v.to_bits())).unwrap_or_default());
    println!("peak value              = {:?}  bits=0x{:08X}", peak, peak.to_bits());
    println!("peak at x               = {:?}  bits=0x{:08X}", peak_x, peak_x.to_bits());

    // Spot-check the exact x claimed:
    let xr = rational(8.442762);
    println!("rational(8.442762)      = {:?}  bits=0x{:08X}", xr, xr.to_bits());
    let x9 = rational(9.0);
    println!("rational(9.0)           = {:?}  bits=0x{:08X}", x9, x9.to_bits());
}
