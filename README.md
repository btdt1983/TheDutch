# Mastering DSP — Rust + WASM starter

Een schone, fase-correcte stereo mastering-engine in Rust, gecompileerd naar
WebAssembly en gebruikt vanuit een React-app via een AudioWorklet. Vervangt de
black-box Web Audio nodes (`DynamicsCompressorNode`, biquads) door eigen DSP
waar je elke coëfficiënt en elke sample latency van kent.

## Wat er nu in zit (v1)

Stereo signaalketen:

```
input  →  M/S width  →  3-band biquad EQ  →  multiband crossover (true-bypass)  →  output gain  →  output
```

- **3-band EQ** — low shelf @ 120 Hz, mid peak @ 1.2 kHz Q=0.9, high shelf @ 8 kHz. RBJ-cookbook biquads, Direct Form I.
- **3-weg multiband** — Linkwitz-Riley 4e orde @ 200 Hz / 2 kHz, met fase-compensatie-allpasses zodat de banden vlak terugsommeren.
- **True-bypass** — wanneer multiband uit staat, gaat het signaal er *helemaal* omheen (geen fase-smearing in je master). Toggle is klik-vrij want de filterstate blijft warm.
- **Parameter-smoothing** — width, gain en wet/dry hebben een 20–50 ms one-pole smoother, dus geen zipper-noise op slider-bewegingen.
- **Stereo throughout** — onafhankelijke filterstate per kanaal.

Niet in v1 (komt later, modulair): per-band compressors, limiter, exciter, meters. De architectuur staat — de rest is "meer van hetzelfde".

## Repostructuur

```
dsp/                              ← Rust crate
  Cargo.toml
  src/lib.rs                      ← alle DSP + C-ABI exports
web/                              ← JS-bestanden voor je React-app
  mastering-processor.js          ← AudioWorklet, draadje los van de UI
  useMasteringEngine.ts           ← React-hook
.github/workflows/
  build-wasm.yml                  ← bouwt .wasm in de cloud
```
