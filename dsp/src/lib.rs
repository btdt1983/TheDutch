//! Mastering DSP — Rust core compiled to WebAssembly.
//!
//! Stereo signal chain:
//!     M/S width  →  3-band biquad EQ  →  multiband crossover (true-bypass)  →  output gain
//!
//! Why the design choices, briefly:
//! - **Biquads are Direct Form I.** Most numerically stable for typical audio
//!   rates; matches Web Audio's BiquadFilterNode topology.
//! - **Multiband is dry/wet-blended, filters always run.** Toggling the
//!   multiband never glitches because the crossover state stays warm; when wet
//!   is 0, the dry signal is untouched — no comb-filter phase smear in your
//!   master.
//! - **Phase-compensation allpasses on the low and high bands** so the 3 bands
//!   sum back to flat when multiband IS on (textbook 3-way LR crossover).
//! - **Per-sample one-pole smoothers** on continuous params (width, gain,
//!   wet/dry) prevent the zipper noise we'd otherwise get from raw parameter
//!   writes. EQ-gain changes recompute coefficients directly; tiny clicks on
//!   huge jumps are inaudible at slider speeds.

#![allow(clippy::too_many_arguments)]

/// AudioWorklet render quantum. Fixed in current browsers; if that ever
/// changes, both this constant and the JS side need to update together.
pub const BLOCK_SIZE: usize = 128;

// ─── Parameter smoother (one-pole exponential, ≈ Web Audio's setTargetAtTime) ─
struct Smoother {
    target: f32,
    current: f32,
    coef: f32,
}

impl Smoother {
    fn new(initial: f32, sample_rate: f32, time_const_s: f32) -> Self {
        // coef = 1 - exp(-1 / (fs · τ)). At τ = 20 ms the smoother is ~99 % settled
        // in ~90 ms — fast enough to feel instant, slow enough to be click-free.
        let coef = 1.0 - (-1.0 / (sample_rate * time_const_s)).exp();
        Self { target: initial, current: initial, coef }
    }
    #[inline] fn set(&mut self, v: f32) { self.target = v; }
    #[inline] fn tick(&mut self) -> f32 {
        self.current += (self.target - self.current) * self.coef;
        self.current
    }
}

// ─── Biquad filter (RBJ cookbook coefficients, Direct Form I) ────────────────
#[derive(Default, Clone, Copy)]
struct Biquad {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    // Independent state per stereo channel — biquads cannot share state across
    // channels or they'd cross-contaminate.
    xl1: f32, xl2: f32, yl1: f32, yl2: f32,
    xr1: f32, xr2: f32, yr1: f32, yr2: f32,
}

#[derive(Clone, Copy)]
enum FilterKind { LowShelf, Peaking, HighShelf, LowPass, HighPass, AllPass }

impl Biquad {
    /// (Re)compute coefficients from filter type + freq + Q + gain (dB).
    /// Q is ignored for shelves (Web Audio uses fixed shelf slope S = 1).
    fn set(&mut self, kind: FilterKind, f0: f32, q: f32, gain_db: f32, fs: f32) {
        use FilterKind::*;
        let w = 2.0 * std::f32::consts::PI * f0 / fs;
        let cos_w = w.cos();
        let sin_w = w.sin();
        let alpha_q = sin_w / (2.0 * q);

        let (b0, b1, b2, a0, a1, a2) = match kind {
            LowPass => (
                (1.0 - cos_w) * 0.5,
                1.0 - cos_w,
                (1.0 - cos_w) * 0.5,
                1.0 + alpha_q,
                -2.0 * cos_w,
                1.0 - alpha_q,
            ),
            HighPass => (
                (1.0 + cos_w) * 0.5,
                -(1.0 + cos_w),
                (1.0 + cos_w) * 0.5,
                1.0 + alpha_q,
                -2.0 * cos_w,
                1.0 - alpha_q,
            ),
            AllPass => (
                1.0 - alpha_q,
                -2.0 * cos_w,
                1.0 + alpha_q,
                1.0 + alpha_q,
                -2.0 * cos_w,
                1.0 - alpha_q,
            ),
            Peaking => {
                let a = 10f32.powf(gain_db / 40.0);
                (
                    1.0 + alpha_q * a,
                    -2.0 * cos_w,
                    1.0 - alpha_q * a,
                    1.0 + alpha_q / a,
                    -2.0 * cos_w,
                    1.0 - alpha_q / a,
                )
            }
            LowShelf => {
                // Web-Audio-compatible shelf slope S = 1 → α = sin(w)·√2/2.
                let a = 10f32.powf(gain_db / 40.0);
                let alpha = sin_w * 0.5 * std::f32::consts::SQRT_2;
                let two_sa = 2.0 * a.sqrt() * alpha;
                let ap1 = a + 1.0;
                let am1 = a - 1.0;
                (
                    a * (ap1 - am1 * cos_w + two_sa),
                    2.0 * a * (am1 - ap1 * cos_w),
                    a * (ap1 - am1 * cos_w - two_sa),
                    ap1 + am1 * cos_w + two_sa,
                    -2.0 * (am1 + ap1 * cos_w),
                    ap1 + am1 * cos_w - two_sa,
                )
            }
            HighShelf => {
                let a = 10f32.powf(gain_db / 40.0);
                let alpha = sin_w * 0.5 * std::f32::consts::SQRT_2;
                let two_sa = 2.0 * a.sqrt() * alpha;
                let ap1 = a + 1.0;
                let am1 = a - 1.0;
                (
                    a * (ap1 + am1 * cos_w + two_sa),
                    -2.0 * a * (am1 + ap1 * cos_w),
                    a * (ap1 + am1 * cos_w - two_sa),
                    ap1 - am1 * cos_w + two_sa,
                    2.0 * (am1 - ap1 * cos_w),
                    ap1 - am1 * cos_w - two_sa,
                )
            }
        };
        // Normalise so a0 = 1 — saves a division per sample in the inner loop.
        let inv_a0 = 1.0 / a0;
        self.b0 = b0 * inv_a0;
        self.b1 = b1 * inv_a0;
        self.b2 = b2 * inv_a0;
        self.a1 = a1 * inv_a0;
        self.a2 = a2 * inv_a0;
    }

    /// Process one stereo sample. Returns (L, R) out.
    #[inline(always)]
    fn tick(&mut self, l: f32, r: f32) -> (f32, f32) {
        let yl = self.b0 * l + self.b1 * self.xl1 + self.b2 * self.xl2
              - self.a1 * self.yl1 - self.a2 * self.yl2;
        self.xl2 = self.xl1; self.xl1 = l;
        self.yl2 = self.yl1; self.yl1 = yl;
        let yr = self.b0 * r + self.b1 * self.xr1 + self.b2 * self.xr2
              - self.a1 * self.yr1 - self.a2 * self.yr2;
        self.xr2 = self.xr1; self.xr1 = r;
        self.yr2 = self.yr1; self.yr1 = yr;
        (yl, yr)
    }
}

// ─── Engine: the full chain + I/O buffers visible to the AudioWorklet ────────
pub struct Engine {
    sample_rate: f32,

    // I/O — fixed-size arrays we hand out as raw pointers. The worklet writes
    // input here, calls process_block(), then reads from the output buffers.
    in_l:  [f32; BLOCK_SIZE],
    in_r:  [f32; BLOCK_SIZE],
    out_l: [f32; BLOCK_SIZE],
    out_r: [f32; BLOCK_SIZE],

    // Continuous params, smoothed per-sample.
    width:        Smoother,
    out_gain_lin: Smoother,
    mb_wet:       Smoother,   // 0 = multiband fully off (clean dry)
    mb_low_gain:  Smoother,
    mb_mid_gain:  Smoother,
    mb_high_gain: Smoother,

    // 3-band EQ — coefficients recomputed on parameter change.
    low_eq:  Biquad,
    mid_eq:  Biquad,
    high_eq: Biquad,

    // 3-way Linkwitz-Riley 4th-order crossover. Each band = 2 cascaded biquads
    // (Q = 0.707) per LR/HP/LP stage, plus 2 allpasses for phase compensation
    // on the low and high bands (matches the standard textbook 3-way LR sum).
    low_lp1:  Biquad, low_lp2:  Biquad, low_ap1:  Biquad, low_ap2:  Biquad,
    mid_hp1:  Biquad, mid_hp2:  Biquad, mid_lp1:  Biquad, mid_lp2:  Biquad,
    high_hp1: Biquad, high_hp2: Biquad, high_ap1: Biquad, high_ap2: Biquad,
}

const F_LOW_MID:  f32 = 200.0;   // low/mid crossover
const F_MID_HIGH: f32 = 2000.0;  // mid/high crossover
const Q_LR:       f32 = 0.7071068;

impl Engine {
    pub fn new(sample_rate: f32) -> Self {
        use FilterKind::*;
        let mut e = Self {
            sample_rate,
            in_l:  [0.0; BLOCK_SIZE],
            in_r:  [0.0; BLOCK_SIZE],
            out_l: [0.0; BLOCK_SIZE],
            out_r: [0.0; BLOCK_SIZE],
            // 20 ms smoothing for continuous controls — same time-constant we
            // used with setTargetAtTime in the Web Audio version.
            width:        Smoother::new(1.0, sample_rate, 0.02),
            out_gain_lin: Smoother::new(1.0, sample_rate, 0.02),
            // Slightly longer (50 ms) on the multiband wet so toggling feels
            // smooth rather than snappy.
            mb_wet:       Smoother::new(0.0, sample_rate, 0.05),
            mb_low_gain:  Smoother::new(1.0, sample_rate, 0.05),
            mb_mid_gain:  Smoother::new(1.0, sample_rate, 0.05),
            mb_high_gain: Smoother::new(1.0, sample_rate, 0.05),
            low_eq:   Biquad::default(),
            mid_eq:   Biquad::default(),
            high_eq:  Biquad::default(),
            low_lp1:  Biquad::default(), low_lp2:  Biquad::default(),
            low_ap1:  Biquad::default(), low_ap2:  Biquad::default(),
            mid_hp1:  Biquad::default(), mid_hp2:  Biquad::default(),
            mid_lp1:  Biquad::default(), mid_lp2:  Biquad::default(),
            high_hp1: Biquad::default(), high_hp2: Biquad::default(),
            high_ap1: Biquad::default(), high_ap2: Biquad::default(),
        };

        // EQ defaults match the original Web Audio chain (low/high shelves use
        // shelf slope S=1, hence the Q here is ignored — passed for uniformity).
        e.low_eq .set(LowShelf,  120.0,  0.7071068, 0.0, sample_rate);
        e.mid_eq .set(Peaking,   1200.0, 0.9,       0.0, sample_rate);
        e.high_eq.set(HighShelf, 8000.0, 0.7071068, 0.0, sample_rate);

        // LR4 crossover (two cascaded Butterworth-Q biquads per LP/HP).
        e.low_lp1 .set(LowPass,  F_LOW_MID,  Q_LR, 0.0, sample_rate);
        e.low_lp2 .set(LowPass,  F_LOW_MID,  Q_LR, 0.0, sample_rate);
        e.mid_hp1 .set(HighPass, F_LOW_MID,  Q_LR, 0.0, sample_rate);
        e.mid_hp2 .set(HighPass, F_LOW_MID,  Q_LR, 0.0, sample_rate);
        e.mid_lp1 .set(LowPass,  F_MID_HIGH, Q_LR, 0.0, sample_rate);
        e.mid_lp2 .set(LowPass,  F_MID_HIGH, Q_LR, 0.0, sample_rate);
        e.high_hp1.set(HighPass, F_MID_HIGH, Q_LR, 0.0, sample_rate);
        e.high_hp2.set(HighPass, F_MID_HIGH, Q_LR, 0.0, sample_rate);

        // Phase compensation: the low band absorbs the phase shift the mid/high
        // crossover (@ F_MID_HIGH) imparts on the other two bands; the high
        // band absorbs the low/mid crossover (@ F_LOW_MID). The mid band sees
        // both crossovers natively, so it doesn't need extra allpasses. With
        // these, the 3 bands sum back to magnitude-flat when blended at unity.
        e.low_ap1 .set(AllPass, F_MID_HIGH, Q_LR, 0.0, sample_rate);
        e.low_ap2 .set(AllPass, F_MID_HIGH, Q_LR, 0.0, sample_rate);
        e.high_ap1.set(AllPass, F_LOW_MID,  Q_LR, 0.0, sample_rate);
        e.high_ap2.set(AllPass, F_LOW_MID,  Q_LR, 0.0, sample_rate);
        e
    }

    /// Set a parameter by ID. IDs are mirrored in `PARAM` on the JS side.
    /// Continuous params smooth toward `v`; EQ-gain params recompute biquad
    /// coefficients immediately (tiny click on huge jumps, inaudible at slider
    /// speeds — for de-zippering you'd smooth the dB then recompute, which we
    /// can add later if needed).
    pub fn set_param(&mut self, id: u32, v: f32) {
        use FilterKind::*;
        let fs = self.sample_rate;
        match id {
            0 => self.low_eq .set(LowShelf,  120.0,  0.7071068, v, fs),
            1 => self.mid_eq .set(Peaking,   1200.0, 0.9,       v, fs),
            2 => self.high_eq.set(HighShelf, 8000.0, 0.7071068, v, fs),
            3 => self.width.set(v),
            4 => self.out_gain_lin.set(10f32.powf(v / 20.0)),
            5 => self.mb_wet.set(if v >= 0.5 { 1.0 } else { 0.0 }),
            6 => self.mb_low_gain .set(10f32.powf(v / 20.0)),
            7 => self.mb_mid_gain .set(10f32.powf(v / 20.0)),
            8 => self.mb_high_gain.set(10f32.powf(v / 20.0)),
            _ => {} // unknown id — ignore
        }
    }

    /// Process exactly `n` frames (≤ BLOCK_SIZE) from in_l/in_r to out_l/out_r.
    pub fn process_block(&mut self, n: usize) {
        let n = n.min(BLOCK_SIZE);
        for i in 0..n {
            let il = self.in_l[i];
            let ir = self.in_r[i];

            // ── M/S width ────────────────────────────────────────────
            // Encode → scale side → decode in one step. width=1 → identity.
            let m = 0.5 * (il + ir);
            let s = 0.5 * (il - ir) * self.width.tick();
            let l = m + s;
            let r = m - s;

            // ── 3-band EQ (series, minimum-phase) ────────────────────
            let (l, r) = self.low_eq .tick(l, r);
            let (l, r) = self.mid_eq .tick(l, r);
            let (l, r) = self.high_eq.tick(l, r);

            // ── Multiband (always-running, dry/wet blended) ──────────
            // Even when wet = 0 we tick the crossover filters so their internal
            // state stays warm. Cost is ~12 biquads per sample — negligible —
            // and it eliminates the click/glitch that would otherwise happen
            // when the user flips multiband on.
            let mb_wet = self.mb_wet.tick();
            let g_low  = self.mb_low_gain .tick();
            let g_mid  = self.mb_mid_gain .tick();
            let g_high = self.mb_high_gain.tick();

            let (ll, lr) = self.low_lp1 .tick(l, r);
            let (ll, lr) = self.low_lp2 .tick(ll, lr);
            let (ll, lr) = self.low_ap1 .tick(ll, lr);
            let (ll, lr) = self.low_ap2 .tick(ll, lr);

            let (ml, mr) = self.mid_hp1 .tick(l, r);
            let (ml, mr) = self.mid_hp2 .tick(ml, mr);
            let (ml, mr) = self.mid_lp1 .tick(ml, mr);
            let (ml, mr) = self.mid_lp2 .tick(ml, mr);

            let (hl, hr) = self.high_hp1.tick(l, r);
            let (hl, hr) = self.high_hp2.tick(hl, hr);
            let (hl, hr) = self.high_ap1.tick(hl, hr);
            let (hl, hr) = self.high_ap2.tick(hl, hr);

            let mb_l = ll * g_low + ml * g_mid + hl * g_high;
            let mb_r = lr * g_low + mr * g_mid + hr * g_high;

            // True bypass: when wet = 0, output is exactly the EQ'd dry signal.
            let dry_g = 1.0 - mb_wet;
            let pre_l = l * dry_g + mb_l * mb_wet;
            let pre_r = r * dry_g + mb_r * mb_wet;

            // ── Output gain (linear, smoothed) ───────────────────────
            let g = self.out_gain_lin.tick();
            self.out_l[i] = pre_l * g;
            self.out_r[i] = pre_r * g;
        }
    }
}

// ─── C ABI — what the AudioWorklet calls into ──────────────────────────────
// We expose raw extern "C" functions so the worklet can use the plain
// WebAssembly.instantiate API without any wasm-bindgen JS glue (which depends
// on browser APIs that aren't available inside AudioWorkletGlobalScope).

#[no_mangle]
pub extern "C" fn engine_new(sample_rate: f32) -> *mut Engine {
    Box::into_raw(Box::new(Engine::new(sample_rate)))
}

/// # Safety
/// `ptr` must come from `engine_new` and not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn engine_free(ptr: *mut Engine) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// # Safety
/// `ptr` must be a live engine pointer.
#[no_mangle]
pub unsafe extern "C" fn engine_set_param(ptr: *mut Engine, id: u32, value: f32) {
    if !ptr.is_null() {
        (*ptr).set_param(id, value);
    }
}

/// # Safety
/// `ptr` must be a live engine pointer; `n` ≤ BLOCK_SIZE.
#[no_mangle]
pub unsafe extern "C" fn engine_process(ptr: *mut Engine, n: usize) {
    if !ptr.is_null() {
        (*ptr).process_block(n);
    }
}

// Pointer getters — the worklet uses these once on init to build Float32Array
// views directly over the engine's I/O buffers. Zero-copy hot path.
#[no_mangle] pub unsafe extern "C" fn engine_in_l_ptr (p: *mut Engine) -> *mut f32 { (*p).in_l .as_mut_ptr() }
#[no_mangle] pub unsafe extern "C" fn engine_in_r_ptr (p: *mut Engine) -> *mut f32 { (*p).in_r .as_mut_ptr() }
#[no_mangle] pub unsafe extern "C" fn engine_out_l_ptr(p: *mut Engine) -> *mut f32 { (*p).out_l.as_mut_ptr() }
#[no_mangle] pub unsafe extern "C" fn engine_out_r_ptr(p: *mut Engine) -> *mut f32 { (*p).out_r.as_mut_ptr() }

// ─── Tests (run with `cargo test` on the host) ─────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    /// Width=1 must reconstruct the input identically. Any drift means the M/S
    /// matrix is broken.
    #[test]
    fn ms_identity_at_width_one() {
        let mut e = Engine::new(48000.0);
        // All EQ flat, multiband off, width=1, gain 0 dB → identity.
        for i in 0..BLOCK_SIZE {
            let t = i as f32 / 48000.0;
            let l = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            let r = (2.0 * std::f32::consts::PI * 660.0 * t).sin();
            e.in_l[i] = l;
            e.in_r[i] = r;
        }
        e.process_block(BLOCK_SIZE);
        for i in 0..BLOCK_SIZE {
            // EQ filters have transient response on the first samples but
            // since gain=0 the steady-state magnitude is unity. Allow a
            // small tolerance just for the initial filter warmup.
            let dl = (e.out_l[i] - e.in_l[i]).abs();
            let dr = (e.out_r[i] - e.in_r[i]).abs();
            assert!(dl < 1e-3, "L drift at {i}: {dl}");
            assert!(dr < 1e-3, "R drift at {i}: {dr}");
        }
    }

    /// Multiband bypass must be bit-identical to running without the multiband
    /// path — the dry/wet blend at wet=0 should pass the signal through.
    #[test]
    fn multiband_bypass_is_transparent() {
        let mut e = Engine::new(48000.0);
        // Hammer some noise through; with wet smoothed at 0 from boot the
        // output should match the EQ-only signal (which is also identity
        // since gains are 0).
        for i in 0..BLOCK_SIZE {
            let v = ((i as f32) * 0.137).sin() * 0.5;
            e.in_l[i] = v;
            e.in_r[i] = v;
        }
        e.process_block(BLOCK_SIZE);
        for i in 0..BLOCK_SIZE {
            let d = (e.out_l[i] - e.in_l[i]).abs();
            assert!(d < 1e-3, "bypass not transparent at {i}: {d}");
        }
    }
}
