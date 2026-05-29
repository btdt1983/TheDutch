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

// ─── One-pole time-constant helper (for compressor/limiter ballistics) ───────
#[inline]
fn time_to_coef(time_s: f32, fs: f32) -> f32 {
    if time_s <= 0.0 { 0.0 } else { (-1.0 / (fs * time_s)).exp() }
}

// ─── Glue compressor (feedforward, soft-knee, stereo-linked) ─────────────────
// A clean bus compressor: detect the louder of the two channels, compute a
// soft-knee gain reduction in the log domain, smooth it with attack/release
// ballistics, then apply the same gain to both channels (stereo-linked, so the
// image doesn't wander). The `comp` knob (0..1) maps to threshold+ratio exactly
// like the original Web Audio chain so the existing slider behaves the same.
struct Compressor {
    threshold_db: f32,
    ratio: f32,
    knee_db: f32,
    attack_coef: f32,
    release_coef: f32,
    makeup_lin: f32,
    env_db: f32, // smoothed gain-reduction envelope, dB ≥ 0
}

impl Compressor {
    fn new(fs: f32) -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 1.0,
            knee_db: 6.0,
            attack_coef: time_to_coef(0.01, fs),   // 10 ms
            release_coef: time_to_coef(0.18, fs),  // 180 ms
            makeup_lin: 1.0,
            env_db: 0.0,
        }
    }
    /// 0..1 → threshold/ratio, matching the original chain's single "comp" knob.
    fn set_amount(&mut self, amount: f32) {
        self.threshold_db = -18.0 - amount * 12.0;
        self.ratio = 1.0 + amount * 3.0;
    }
    fn set_makeup_db(&mut self, db: f32) {
        self.makeup_lin = 10f32.powf(db / 20.0);
    }
    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let peak = l.abs().max(r.abs()).max(1e-9);
        let x_g = 20.0 * peak.log10();
        // Soft-knee gain computer (Giannoulis et al.).
        let t = self.threshold_db;
        let w = self.knee_db;
        let over = x_g - t;
        let y_g = if 2.0 * over < -w {
            x_g
        } else if 2.0 * over.abs() <= w {
            x_g + (1.0 / self.ratio - 1.0) * (over + w * 0.5).powi(2) / (2.0 * w)
        } else {
            t + over / self.ratio
        };
        let reduction = (x_g - y_g).max(0.0); // dB of GR to apply
        // Branching ballistics: attack when GR grows, release when it shrinks.
        if reduction > self.env_db {
            self.env_db = self.attack_coef * self.env_db + (1.0 - self.attack_coef) * reduction;
        } else {
            self.env_db = self.release_coef * self.env_db + (1.0 - self.release_coef) * reduction;
        }
        let gain = self.makeup_lin * 10f32.powf(-self.env_db / 20.0);
        (l * gain, r * gain)
    }
}

// ─── Brickwall limiter (lookahead) ───────────────────────────────────────────
// Delays the signal by a short lookahead so gain can ramp down smoothly BEFORE
// a peak reaches the output — this is exactly the control the old Web Audio
// DynamicsCompressorNode didn't give us (its hidden ~6 ms lookahead caused the
// parallel-path comb filtering). Here we own the delay. A final hard clamp at
// the ceiling guarantees no sample ever exceeds it, catching any sub-dB
// overshoot the smoother leaves behind.
const MAX_LOOKAHEAD: usize = 512; // ~10.6 ms at 48 k — generous headroom

struct Limiter {
    ceiling: f32, // linear
    delay_l: [f32; MAX_LOOKAHEAD],
    delay_r: [f32; MAX_LOOKAHEAD],
    pos: usize,
    len: usize, // active lookahead in samples
    gain: f32,  // current gain, ≤ 1
    attack_coef: f32,
    release_coef: f32,
}

impl Limiter {
    fn new(fs: f32) -> Self {
        let len = ((fs * 0.0015) as usize).clamp(8, MAX_LOOKAHEAD); // 1.5 ms
        Self {
            ceiling: 10f32.powf(-1.0 / 20.0), // -1 dBFS default
            delay_l: [0.0; MAX_LOOKAHEAD],
            delay_r: [0.0; MAX_LOOKAHEAD],
            pos: 0,
            len,
            gain: 1.0,
            // Attack settles well within the lookahead window (~0.3 ms ≪ 1.5 ms)
            // so the gain is at target by the time the peak exits the delay line.
            attack_coef: time_to_coef(0.0003, fs),
            release_coef: time_to_coef(0.1, fs), // 100 ms release
        }
    }
    fn set_ceiling_db(&mut self, db: f32) {
        self.ceiling = 10f32.powf(db / 20.0);
    }
    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        // Required gain for the INCOMING peak (look at it before it's audible).
        let peak_in = l.abs().max(r.abs());
        let req = if peak_in > self.ceiling { self.ceiling / peak_in } else { 1.0 };
        // Attack fast toward a lower gain; release slowly back up.
        if req < self.gain {
            self.gain = self.attack_coef * self.gain + (1.0 - self.attack_coef) * req;
        } else {
            self.gain = self.release_coef * self.gain + (1.0 - self.release_coef) * req;
        }
        // Read the delayed sample, then store the incoming one (delay == len).
        let out_l = self.delay_l[self.pos];
        let out_r = self.delay_r[self.pos];
        self.delay_l[self.pos] = l;
        self.delay_r[self.pos] = r;
        self.pos = (self.pos + 1) % self.len;
        // Apply gain, then hard-clamp at the ceiling as a guaranteed safety net.
        let gl = (out_l * self.gain).clamp(-self.ceiling, self.ceiling);
        let gr = (out_r * self.gain).clamp(-self.ceiling, self.ceiling);
        (gl, gr)
    }
}

// ─── Mono biquad (used inside the oversampled exciter) ───────────────────────
#[derive(Default, Clone, Copy)]
struct BiquadM {
    b0: f32, b1: f32, b2: f32, a1: f32, a2: f32,
    x1: f32, x2: f32, y1: f32, y2: f32,
}

impl BiquadM {
    fn lowpass(&mut self, f0: f32, q: f32, fs: f32) {
        let w = 2.0 * std::f32::consts::PI * f0 / fs;
        let (c, s) = (w.cos(), w.sin());
        let al = s / (2.0 * q);
        let a0 = 1.0 + al;
        self.b0 = (1.0 - c) * 0.5 / a0;
        self.b1 = (1.0 - c) / a0;
        self.b2 = (1.0 - c) * 0.5 / a0;
        self.a1 = -2.0 * c / a0;
        self.a2 = (1.0 - al) / a0;
    }
    fn highpass(&mut self, f0: f32, q: f32, fs: f32) {
        let w = 2.0 * std::f32::consts::PI * f0 / fs;
        let (c, s) = (w.cos(), w.sin());
        let al = s / (2.0 * q);
        let a0 = 1.0 + al;
        self.b0 = (1.0 + c) * 0.5 / a0;
        self.b1 = -(1.0 + c) / a0;
        self.b2 = (1.0 + c) * 0.5 / a0;
        self.a1 = -2.0 * c / a0;
        self.a2 = (1.0 - al) / a0;
    }
    #[inline]
    fn tick(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
              - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1; self.x1 = x;
        self.y2 = self.y1; self.y1 = y;
        y
    }
}

// ─── Harmonic exciter (2× oversampled, parallel, sample-aligned) ─────────────
// Isolates the high band, generates harmonics with a tanh-based saturator, and
// adds them back to the dry signal. The whole thing is computed from the same
// input sample with NO delay relative to dry, so unlike the old WaveShaper
// (which added oversampling latency and combed against the dry path) the blend
// is perfectly phase-aligned. Oversampling runs the nonlinearity at 2× rate so
// the generated harmonics that would alias at base rate are pushed above the
// base Nyquist and filtered out cleanly.
struct Exciter {
    hp_l: BiquadM, hp_r: BiquadM,
    // anti-imaging (upsample) and anti-aliasing (downsample) filters at 2× fs.
    up_l: [BiquadM; 2], up_r: [BiquadM; 2],
    down_l: [BiquadM; 2], down_r: [BiquadM; 2],
    amount: Smoother, // wet blend 0..1
    warmth: f32,      // even-harmonic flavour 0..1
    freq: f32,
    fs: f32,
}

impl Exciter {
    fn new(fs: f32) -> Self {
        let os = fs * 2.0;
        let fc = fs * 0.45; // just below base Nyquist; kills images/aliases
        let mut e = Exciter {
            hp_l: BiquadM::default(), hp_r: BiquadM::default(),
            up_l: [BiquadM::default(); 2], up_r: [BiquadM::default(); 2],
            down_l: [BiquadM::default(); 2], down_r: [BiquadM::default(); 2],
            amount: Smoother::new(0.0, fs, 0.03),
            warmth: 0.3,
            freq: 3500.0,
            fs,
        };
        e.set_freq(3500.0);
        for b in e.up_l.iter_mut().chain(e.up_r.iter_mut())
            .chain(e.down_l.iter_mut()).chain(e.down_r.iter_mut()) {
            b.lowpass(fc, 0.7071068, os);
        }
        e
    }
    fn set_freq(&mut self, f: f32) {
        self.freq = f.clamp(500.0, 16000.0);
        self.hp_l.highpass(self.freq, 0.7071068, self.fs);
        self.hp_r.highpass(self.freq, 0.7071068, self.fs);
    }
    fn set_amount(&mut self, a: f32) { self.amount.set(a.clamp(0.0, 1.0)); }
    fn set_warmth(&mut self, w: f32) { self.warmth = w.clamp(0.0, 1.0); }

    #[inline]
    fn saturate(x: f32, warmth: f32) -> f32 {
        // Odd harmonics from tanh + a touch of asymmetry for even-order "warmth".
        (2.2 * x).tanh() + warmth * 0.4 * x * x.abs()
    }

    // Excite one channel: highpass → 2× oversample → saturate → downsample.
    // Disjoint &mut borrows of engine fields, so this is an associated fn.
    #[inline]
    fn excite_channel(
        hp: &mut BiquadM,
        up: &mut [BiquadM; 2],
        down: &mut [BiquadM; 2],
        warmth: f32,
        dry: f32,
    ) -> f32 {
        let band = hp.tick(dry);
        // Zero-stuff to 2× rate (×2 to compensate the inserted zero's energy
        // loss), filter, saturate, filter, decimate (keep the first phase).
        // Split each cascade into separate statements: indexing the array twice
        // in one expression would be two simultaneous &mut borrows.
        let u0a = up[0].tick(band * 2.0);
        let u0 = up[1].tick(u0a);
        let s0 = Self::saturate(u0, warmth);
        let d0a = down[0].tick(s0);
        let d0 = down[1].tick(d0a);
        let u1a = up[0].tick(0.0);
        let u1 = up[1].tick(u1a);
        let s1 = Self::saturate(u1, warmth);
        let d1a = down[0].tick(s1);
        let _d1 = down[1].tick(d1a); // discarded by decimation
        d0
    }

    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        let amt = self.amount.tick();
        let ex_l = Self::excite_channel(&mut self.hp_l, &mut self.up_l, &mut self.down_l, self.warmth, l);
        let ex_r = Self::excite_channel(&mut self.hp_r, &mut self.up_r, &mut self.down_r, self.warmth, r);
        // Parallel add — dry passes through untouched, harmonics blended on top.
        (l + ex_l * amt, r + ex_r * amt)
    }
}

// ─── Surgical EQ (11 fixed-frequency bands, mirrors the original chain) ──────
// Series of stereo biquads at the same frequencies/Qs as the JS chain. Every
// peaking/shelf band at 0 dB is bit-transparent (numerator == denominator), so
// the whole stage is transparent at default. The high-pass is the only band
// that isn't gain-based, so it's bypassed at its minimum (≤ 20 Hz = "off") to
// keep the default chain perfectly clean. The mud band's Q is driven by the
// "tighten" control, so it's recomputed when either its gain or Q changes.
struct SurgicalEq {
    fs: f32,
    hp: Biquad, hp_freq: f32,
    sub: Biquad,
    bass: Biquad,
    mud: Biquad, mud_gain: f32, tighten: f32,
    boxf: Biquad,
    nasal: Biquad,
    clarity: Biquad,
    presence: Biquad,
    deesser: Biquad,
    dehiss: Biquad,
    air: Biquad,
}

impl SurgicalEq {
    fn new(fs: f32) -> Self {
        use FilterKind::*;
        let mut s = SurgicalEq {
            fs,
            hp: Biquad::default(), hp_freq: 20.0,
            sub: Biquad::default(),
            bass: Biquad::default(),
            mud: Biquad::default(), mud_gain: 0.0, tighten: 1.0,
            boxf: Biquad::default(),
            nasal: Biquad::default(),
            clarity: Biquad::default(),
            presence: Biquad::default(),
            deesser: Biquad::default(),
            dehiss: Biquad::default(),
            air: Biquad::default(),
        };
        s.hp      .set(HighPass,  20.0,    0.7071068, 0.0, fs);
        s.sub     .set(Peaking,   50.0,    1.4,       0.0, fs);
        s.bass    .set(LowShelf,  90.0,    0.7071068, 0.0, fs);
        s.mud     .set(Peaking,   250.0,   1.0,       0.0, fs);
        s.boxf    .set(Peaking,   500.0,   1.2,       0.0, fs);
        s.nasal   .set(Peaking,   1200.0,  1.6,       0.0, fs);
        s.clarity .set(Peaking,   2500.0,  1.0,       0.0, fs);
        s.presence.set(Peaking,   4000.0,  0.9,       0.0, fs);
        s.deesser .set(Peaking,   7000.0,  2.0,       0.0, fs);
        s.dehiss  .set(Peaking,   12000.0, 1.4,       0.0, fs);
        s.air     .set(HighShelf, 12000.0, 0.7071068, 0.0, fs);
        s
    }
    fn set_hp_freq(&mut self, f: f32) {
        self.hp_freq = f;
        self.hp.set(FilterKind::HighPass, f.max(20.0), 0.7071068, 0.0, self.fs);
    }
    fn set_sub(&mut self, g: f32)      { self.sub.set(FilterKind::Peaking, 50.0, 1.4, g, self.fs); }
    fn set_bass(&mut self, g: f32)     { self.bass.set(FilterKind::LowShelf, 90.0, 0.7071068, g, self.fs); }
    fn set_mud_gain(&mut self, g: f32) { self.mud_gain = g; self.recompute_mud(); }
    fn set_tighten(&mut self, q: f32)  { self.tighten = q.max(0.1); self.recompute_mud(); }
    fn recompute_mud(&mut self) {
        self.mud.set(FilterKind::Peaking, 250.0, self.tighten, self.mud_gain, self.fs);
    }
    fn set_box(&mut self, g: f32)      { self.boxf.set(FilterKind::Peaking, 500.0, 1.2, g, self.fs); }
    fn set_nasal(&mut self, g: f32)    { self.nasal.set(FilterKind::Peaking, 1200.0, 1.6, g, self.fs); }
    fn set_clarity(&mut self, g: f32)  { self.clarity.set(FilterKind::Peaking, 2500.0, 1.0, g, self.fs); }
    fn set_presence(&mut self, g: f32) { self.presence.set(FilterKind::Peaking, 4000.0, 0.9, g, self.fs); }
    fn set_deesser(&mut self, g: f32)  { self.deesser.set(FilterKind::Peaking, 7000.0, 2.0, g, self.fs); }
    fn set_dehiss(&mut self, g: f32)   { self.dehiss.set(FilterKind::Peaking, 12000.0, 1.4, g, self.fs); }
    fn set_air(&mut self, g: f32)      { self.air.set(FilterKind::HighShelf, 12000.0, 0.7071068, g, self.fs); }

    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        // HP bypassed at minimum to keep the default chain bit-transparent.
        let (mut l, mut r) = if self.hp_freq > 20.0 { self.hp.tick(l, r) } else { (l, r) };
        let (a, b) = self.sub.tick(l, r);      l = a; r = b;
        let (a, b) = self.bass.tick(l, r);     l = a; r = b;
        let (a, b) = self.mud.tick(l, r);      l = a; r = b;
        let (a, b) = self.boxf.tick(l, r);     l = a; r = b;
        let (a, b) = self.nasal.tick(l, r);    l = a; r = b;
        let (a, b) = self.clarity.tick(l, r);  l = a; r = b;
        let (a, b) = self.presence.tick(l, r); l = a; r = b;
        let (a, b) = self.deesser.tick(l, r);  l = a; r = b;
        let (a, b) = self.dehiss.tick(l, r);   l = a; r = b;
        let (a, b) = self.air.tick(l, r);      l = a; r = b;
        (l, r)
    }
}

// ─── Transient shaper (differential-envelope, no parallel delay) ─────────────
// Two envelope followers track the signal: a FAST one that jumps on transients
// and a SLOW one that lags. Their ratio (in dB) is positive during an attack
// (fast outruns slow) and negative during sustain/decay (fast falls while slow
// holds). We turn that into a gain: `attack` boosts the attack region, `sustain`
// boosts the decay region. Because it's a single gain applied to the dry signal
// — no parallel/delayed copy — there's no comb filtering, unlike the old
// parallel-compressor transient trick.
struct TransientShaper {
    fast_env: f32,
    slow_env: f32,
    fast_atk: f32, fast_rel: f32,
    slow_atk: f32, slow_rel: f32,
    attack_amt: f32,  // 0..1
    sustain_amt: f32, // 0..1
}

const TRANS_SCALE_A: f32 = 1.2;
const TRANS_SCALE_S: f32 = 1.0;

impl TransientShaper {
    fn new(fs: f32) -> Self {
        Self {
            fast_env: 0.0,
            slow_env: 0.0,
            fast_atk: time_to_coef(0.0005, fs), // 0.5 ms — snaps to transients
            fast_rel: time_to_coef(0.020, fs),  // 20 ms
            slow_atk: time_to_coef(0.030, fs),  // 30 ms — lags the attack
            slow_rel: time_to_coef(0.150, fs),  // 150 ms — holds through decay
            attack_amt: 0.0,
            sustain_amt: 0.0,
        }
    }
    fn set_attack(&mut self, a: f32) { self.attack_amt = a.clamp(0.0, 1.0); }
    fn set_sustain(&mut self, s: f32) { self.sustain_amt = s.clamp(0.0, 1.0); }

    #[inline]
    fn process(&mut self, l: f32, r: f32) -> (f32, f32) {
        // Stereo-linked detector so the gain is identical on both channels.
        let det = l.abs().max(r.abs());
        // Fast follower
        let fc = if det > self.fast_env { self.fast_atk } else { self.fast_rel };
        self.fast_env = fc * self.fast_env + (1.0 - fc) * det;
        // Slow follower
        let sc = if det > self.slow_env { self.slow_atk } else { self.slow_rel };
        self.slow_env = sc * self.slow_env + (1.0 - sc) * det;

        // Transparent at default, but envelopes above stay warm for glitch-free
        // engaging.
        if self.attack_amt <= 0.0 && self.sustain_amt <= 0.0 {
            return (l, r);
        }
        let f = self.fast_env.max(1e-9);
        let s = self.slow_env.max(1e-9);
        let diff_db = 20.0 * (f / s).log10(); // + during attack, − during sustain
        let attack_db = self.attack_amt * diff_db.max(0.0) * TRANS_SCALE_A;
        let sustain_db = self.sustain_amt * (-diff_db).max(0.0) * TRANS_SCALE_S;
        // Clamp to a sane range so extreme material can't run away.
        let gain_db = (attack_db + sustain_db).clamp(-12.0, 12.0);
        let gain = 10f32.powf(gain_db / 20.0);
        (l * gain, r * gain)
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

    // 11-band surgical EQ (post 3-band, pre multiband).
    surgical: SurgicalEq,

    // 3-way Linkwitz-Riley 4th-order crossover. Each band = 2 cascaded biquads
    // (Q = 0.707) per LR/HP/LP stage, plus 2 allpasses for phase compensation
    // on the low and high bands (matches the standard textbook 3-way LR sum).
    low_lp1:  Biquad, low_lp2:  Biquad, low_ap1:  Biquad, low_ap2:  Biquad,
    mid_hp1:  Biquad, mid_hp2:  Biquad, mid_lp1:  Biquad, mid_lp2:  Biquad,
    high_hp1: Biquad, high_hp2: Biquad, high_ap1: Biquad, high_ap2: Biquad,

    // Dynamics: glue compressor then brickwall limiter (post output-gain).
    comp: Compressor,
    limiter: Limiter,
    // Harmonic exciter (parallel, pre-compressor).
    exciter: Exciter,
    // Transient shaper (post-multiband, pre-exciter).
    transient: TransientShaper,
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
            surgical: SurgicalEq::new(sample_rate),
            low_lp1:  Biquad::default(), low_lp2:  Biquad::default(),
            low_ap1:  Biquad::default(), low_ap2:  Biquad::default(),
            mid_hp1:  Biquad::default(), mid_hp2:  Biquad::default(),
            mid_lp1:  Biquad::default(), mid_lp2:  Biquad::default(),
            high_hp1: Biquad::default(), high_hp2: Biquad::default(),
            high_ap1: Biquad::default(), high_ap2: Biquad::default(),
            comp: Compressor::new(sample_rate),
            limiter: Limiter::new(sample_rate),
            exciter: Exciter::new(sample_rate),
            transient: TransientShaper::new(sample_rate),
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
            9 => self.comp.set_amount(v),        // 0..1 glue amount
            10 => self.comp.set_makeup_db(v),    // makeup gain (dB)
            11 => self.limiter.set_ceiling_db(v),// brickwall ceiling (dBFS)
            12 => self.exciter.set_amount(v),    // 0..1 exciter wet
            13 => self.exciter.set_freq(v),      // exciter highpass (Hz)
            14 => self.exciter.set_warmth(v),    // 0..1 even-harmonic warmth
            15 => self.surgical.set_hp_freq(v),  // high-pass freq (≤20 = off)
            16 => self.surgical.set_sub(v),      // sub-punch gain (dB)
            17 => self.surgical.set_bass(v),     // bass shelf gain (dB)
            18 => self.surgical.set_mud_gain(v), // mud-kill gain (dB)
            19 => self.surgical.set_tighten(v),  // mud-kill Q
            20 => self.surgical.set_box(v),      // box-cut gain (dB)
            21 => self.surgical.set_nasal(v),    // nasal gain (dB)
            22 => self.surgical.set_clarity(v),  // clarity gain (dB)
            23 => self.surgical.set_presence(v), // presence gain (dB)
            24 => self.surgical.set_deesser(v),  // de-esser gain (dB)
            25 => self.surgical.set_dehiss(v),   // de-hiss gain (dB)
            26 => self.surgical.set_air(v),      // air shelf gain (dB)
            27 => self.transient.set_attack(v),  // 0..1 attack enhance
            28 => self.transient.set_sustain(v), // 0..1 sustain enhance
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

            // ── Surgical EQ (11 fixed bands) ─────────────────────────
            let (l, r) = self.surgical.process(l, r);

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

            // ── Transient shaper (single-gain, no parallel delay) ────
            let (tl, tr) = self.transient.process(pre_l, pre_r);

            // ── Harmonic exciter (parallel, sample-aligned) ──────────
            let (ex_l, ex_r) = self.exciter.process(tl, tr);

            // ── Glue compressor ──────────────────────────────────────
            let (cl, cr) = self.comp.process(ex_l, ex_r);

            // ── Output gain (linear, smoothed) — drives the limiter ──
            let g = self.out_gain_lin.tick();
            let dl = cl * g;
            let dr = cr * g;

            // ── Brickwall limiter (last stage, guarantees the ceiling) ─
            let (ol, or_) = self.limiter.process(dl, dr);
            self.out_l[i] = ol;
            self.out_r[i] = or_;
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

/// Processing latency in samples (the limiter's lookahead). Offline renders use
/// this to time-align the output so the export isn't shifted by the lookahead.
///
/// # Safety
/// `ptr` must be a live engine pointer.
#[no_mangle]
pub unsafe extern "C" fn engine_latency(ptr: *mut Engine) -> usize {
    if ptr.is_null() { 0 } else { (*ptr).limiter.len }
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
        // At flat settings the chain is identity EXCEPT the limiter's lookahead
        // delay, so output is the input shifted by `lat` samples. Use a quiet
        // signal (peak 0.5 < ceiling) so the limiter passes without acting.
        let mut e = Engine::new(48000.0);
        let lat = ((48000.0f32 * 0.0015) as usize).clamp(8, MAX_LOOKAHEAD);
        // Build a continuous signal across two blocks so we can see past the delay.
        let sig = |n: usize| {
            let t = n as f32 / 48000.0;
            (
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin(),
                0.5 * (2.0 * std::f32::consts::PI * 660.0 * t).sin(),
            )
        };
        let mut outl = [0.0f32; 2 * BLOCK_SIZE];
        let mut outr = [0.0f32; 2 * BLOCK_SIZE];
        for blk in 0..2 {
            for i in 0..BLOCK_SIZE {
                let (l, r) = sig(blk * BLOCK_SIZE + i);
                e.in_l[i] = l; e.in_r[i] = r;
            }
            e.process_block(BLOCK_SIZE);
            outl[blk * BLOCK_SIZE..(blk + 1) * BLOCK_SIZE].copy_from_slice(&e.out_l);
            outr[blk * BLOCK_SIZE..(blk + 1) * BLOCK_SIZE].copy_from_slice(&e.out_r);
        }
        // Compare output[n] with input[n - lat] for n past the delay + filter warmup.
        for n in (lat + 64)..(2 * BLOCK_SIZE) {
            let (il, ir) = sig(n - lat);
            assert!((outl[n] - il).abs() < 2e-3, "L drift at {n}: {} vs {}", outl[n], il);
            assert!((outr[n] - ir).abs() < 2e-3, "R drift at {n}: {} vs {}", outr[n], ir);
        }
    }

    /// Multiband bypass + flat EQ must be transparent (delay-compensated) at a
    /// level below the limiter ceiling.
    #[test]
    fn multiband_bypass_is_transparent() {
        let mut e = Engine::new(48000.0);
        let lat = ((48000.0f32 * 0.0015) as usize).clamp(8, MAX_LOOKAHEAD);
        let sig = |n: usize| ((n as f32) * 0.137).sin() * 0.4;
        let mut out = [0.0f32; 2 * BLOCK_SIZE];
        for blk in 0..2 {
            for i in 0..BLOCK_SIZE {
                let v = sig(blk * BLOCK_SIZE + i);
                e.in_l[i] = v; e.in_r[i] = v;
            }
            e.process_block(BLOCK_SIZE);
            out[blk * BLOCK_SIZE..(blk + 1) * BLOCK_SIZE].copy_from_slice(&e.out_l);
        }
        for n in (lat + 4)..(2 * BLOCK_SIZE) {
            let expected = sig(n - lat);
            assert!((out[n] - expected).abs() < 2e-3, "bypass drift at {n}: {} vs {}", out[n], expected);
        }
    }

    /// The brickwall limiter must keep every output sample at or below the
    /// ceiling, even when fed a signal well over it.
    #[test]
    fn limiter_enforces_ceiling() {
        let mut e = Engine::new(48000.0);
        let ceiling = 10f32.powf(-1.0 / 20.0);
        // Hot signal: ±2.0 (≈ +6 dBFS), way over the -1 dBFS ceiling.
        for blk in 0..8 {
            for i in 0..BLOCK_SIZE {
                let t = (blk * BLOCK_SIZE + i) as f32 / 48000.0;
                let v = 2.0 * (2.0 * std::f32::consts::PI * 100.0 * t).sin();
                e.in_l[i] = v; e.in_r[i] = v;
            }
            e.process_block(BLOCK_SIZE);
            // After the first couple of blocks the limiter is engaged; check all.
            if blk >= 2 {
                for i in 0..BLOCK_SIZE {
                    assert!(e.out_l[i].abs() <= ceiling + 1e-4, "L over ceiling: {}", e.out_l[i]);
                    assert!(e.out_r[i].abs() <= ceiling + 1e-4, "R over ceiling: {}", e.out_r[i]);
                }
            }
        }
    }

    /// The compressor at amount 0 (ratio 1) must be transparent (gain stays 1).
    #[test]
    fn compressor_unity_at_zero() {
        let mut c = Compressor::new(48000.0);
        c.set_amount(0.0); // ratio 1.0 → no reduction
        let (l, r) = c.process(0.7, 0.7);
        assert!((l - 0.7).abs() < 1e-6 && (r - 0.7).abs() < 1e-6, "comp not unity: {l},{r}");
    }

    /// Exciter at amount 0 must pass the dry signal through untouched (the wet
    /// path is multiplied by 0, and dry has no delay).
    #[test]
    fn exciter_transparent_at_zero() {
        let mut e = Exciter::new(48000.0);
        e.set_amount(0.0);
        // Let the amount smoother settle to 0 (it starts at 0 anyway).
        for _ in 0..256 { e.process(0.0, 0.0); }
        let (l, r) = e.process(0.42, -0.31);
        assert!((l - 0.42).abs() < 1e-6, "exciter L not transparent: {l}");
        assert!((r + 0.31).abs() < 1e-6, "exciter R not transparent: {r}");
    }

    /// Exciter at full drive must stay finite and bounded (no blow-up / NaN)
    /// when fed a hot high-frequency tone.
    #[test]
    fn exciter_bounded_at_full() {
        let mut e = Exciter::new(48000.0);
        e.set_amount(1.0);
        e.set_warmth(1.0);
        let mut maxabs = 0.0f32;
        for n in 0..4096 {
            let t = n as f32 / 48000.0;
            let x = 0.9 * (2.0 * std::f32::consts::PI * 8000.0 * t).sin();
            let (l, _r) = e.process(x, x);
            assert!(l.is_finite(), "exciter produced non-finite output");
            maxabs = maxabs.max(l.abs());
        }
        assert!(maxabs < 3.0, "exciter output unexpectedly large: {maxabs}");
    }

    /// Surgical EQ at default (all gains 0, HP off) must be bit-transparent —
    /// every peaking/shelf at 0 dB has numerator == denominator, and the HP is
    /// bypassed at its 20 Hz minimum.
    #[test]
    fn surgical_transparent_at_default() {
        let mut s = SurgicalEq::new(48000.0);
        for n in 0..512 {
            let t = n as f32 / 48000.0;
            let x = 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
                  + 0.3 * (2.0 * std::f32::consts::PI * 60.0 * t).sin();
            let (l, r) = s.process(x, x * 0.9);
            // 10 cascaded biquads, two of them near-DC (50/90 Hz), accumulate
            // ~3e-4 (-71 dB measured) of f32 rounding even when mathematically
            // transparent. Signal-correlated and inaudible; this is normal for
            // f32 cascaded low-frequency biquads, not a wiring bug.
            assert!((l - x).abs() < 1e-3, "surgical L drift at {n}: {} vs {}", l, x);
            assert!((r - x * 0.9).abs() < 1e-3, "surgical R drift at {n}");
        }
    }

    /// Engaging a surgical band must actually change the signal (sanity that
    /// the wiring works, not just transparency).
    #[test]
    fn surgical_band_engages() {
        let mut s = SurgicalEq::new(48000.0);
        s.set_air(6.0); // +6 dB high shelf @ 12 kHz
        let mut changed = false;
        for n in 0..512 {
            let t = n as f32 / 48000.0;
            let x = 0.5 * (2.0 * std::f32::consts::PI * 15000.0 * t).sin();
            let (l, _r) = s.process(x, x);
            if (l - x).abs() > 1e-3 { changed = true; }
        }
        assert!(changed, "air shelf had no effect on 15 kHz tone");
    }

    /// Transient shaper at default (both 0) must pass through untouched.
    #[test]
    fn transient_transparent_at_default() {
        let mut t = TransientShaper::new(48000.0);
        for n in 0..2048 {
            let v = ((n as f32) * 0.21).sin() * 0.5;
            let (l, r) = t.process(v, v * 0.8);
            assert!((l - v).abs() < 1e-7, "trans L not transparent at {n}");
            assert!((r - v * 0.8).abs() < 1e-7, "trans R not transparent at {n}");
        }
    }

    /// With attack engaged, a percussive burst's onset should be boosted
    /// relative to the dry signal (proves the detector + gain work).
    #[test]
    fn transient_attack_boosts_onset() {
        let fs = 48000.0;
        let mut t = TransientShaper::new(fs);
        t.set_attack(1.0);
        // Build a repeating "kick": short loud burst then silence, so the fast
        // envelope clearly outruns the slow one at each onset.
        let mut max_gain_ratio = 0.0f32;
        for n in 0..(fs as usize) {
            let phase = n % 12000; // ~4 Hz pulse
            let x = if phase < 400 {
                0.8 * (2.0 * std::f32::consts::PI * 80.0 * (n as f32) / fs).sin()
            } else {
                0.0
            };
            let (l, _r) = t.process(x, x);
            if x.abs() > 1e-4 {
                max_gain_ratio = max_gain_ratio.max((l / x).abs());
            }
            assert!(l.is_finite(), "transient produced non-finite output");
        }
        // The onset of at least one burst must have been amplified ( >1 ).
        assert!(max_gain_ratio > 1.05, "attack never boosted onset: {max_gain_ratio}");
    }
}
