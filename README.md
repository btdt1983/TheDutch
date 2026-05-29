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

## Opzetten (vanaf je telefoon)

### 1. GitHub-repo maken en deze bestanden erin zetten

Maak een **publieke** repo (zo blijven Actions gratis en onbeperkt). Drag-and-drop deze map via de GitHub-website of de GitHub mobile-app. Belangrijk: behoud de structuur (`dsp/`, `web/`, `.github/workflows/`).

### 2. Wachten tot Actions klaar is

Bij elke push naar `dsp/` draait de workflow automatisch:
- compileert naar `wasm32-unknown-unknown`,
- draait de native unit-tests,
- optimaliseert met `wasm-opt -O3`,
- uploadt `mastering_dsp.wasm` als download-artifact.

Eerste build duurt ~1 min, daarna ~10 s door caching.

### 3. WASM downloaden en in je Lovable/React-app zetten

Ga naar het laatste workflow-resultaat → "Artifacts" → download `mastering-dsp-wasm`. Daarin zit `mastering_dsp.wasm`.

Plaats:
- `mastering_dsp.wasm` in je React-app's `public/` map → wordt geserveerd op `/mastering_dsp.wasm`.
- `web/mastering-processor.js` ook in `public/` → wordt geserveerd op `/mastering-processor.js`.
- `web/useMasteringEngine.ts` in `src/` (of waar je hooks staan).

### 4. Gebruiken in je app

```tsx
import { useMasteringEngine, PARAM } from "./useMasteringEngine";

function MasteringApp() {
  const { ctx, node, setParam, ready, error } = useMasteringEngine({
    wasmUrl: "/mastering_dsp.wasm",
    processorUrl: "/mastering-processor.js",
  });

  const [low, setLow] = useState(0);

  // Hook the slider straight to the engine
  useEffect(() => {
    if (ready) setParam(PARAM.LOW, low);
  }, [low, ready, setParam]);

  const playFile = async (file: File) => {
    if (!ctx || !node) return;
    const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(node);          // <-- into the Rust engine
    source.start();
  };

  if (error) return <div>Engine failed: {error}</div>;
  if (!ready) return <div>Loading engine…</div>;

  return (
    <>
      <input type="file" onChange={(e) => playFile(e.target.files![0])} />
      <input type="range" min={-12} max={12} step={0.1}
             value={low} onChange={(e) => setLow(+e.target.value)} />
    </>
  );
}
```

## Belangrijk over hosting

- **Geen COOP/COEP-headers nodig.** We gebruiken geen SharedArrayBuffer; gewone HTTPS is voldoende.
- **MIME-type van de wasm** wordt door Vite/Next/etc. automatisch goed gezet. Voor exotische hosts: zorg dat `.wasm` als `application/wasm` geserveerd wordt.
- **Lovable + dit project**: Lovable raakt de bestanden in `/public` en `/src` niet zomaar aan zolang ze geen JSX zijn. De `.wasm` en `mastering-processor.js` zijn veilig. Mocht Lovable je `useMasteringEngine.ts` ooit overschrijven, dan importeer je 'm gewoon opnieuw uit deze repo.

## Verifiëren dat het werkt

De Rust-code heeft twee unit-tests die de DSP-correctheid bewijzen:
- `ms_identity_at_width_one` — bij width=1 reconstrueert de M/S-keten de input exact (geen lekkage in stereoveld).
- `multiband_bypass_is_transparent` — wanneer multiband uit staat is de output bit-identiek aan de input minus de EQ (die op 0 dB ook identiek is).

De Actions-workflow draait die tests bij elke build; als er ooit een DSP-regressie sluipt, ziet het rode kruis je dat direct.

## Daarna verder bouwen

Logische volgende blokken, in oplopende complexiteit:

1. **Output limiter** — single-band feedforward, simpel toe te voegen.
2. **Per-band compressors** — in de multiband. Lookahead is optioneel; zonder lookahead vermijden we de latency-mismatch-bugs van de oude Web Audio chain volledig.
3. **Saturator/exciter** — band-limited tanh of soft-clip, oversample met polyphase FIR (geen latency-mismatch met dry omdat we het zelf doen).
4. **Meters** — LUFS-K (BS.1770), true-peak (4× oversampled), allemaal in dezelfde WASM zodat het exporteerbaar deterministisch is.

Elke uitbreiding voegt een nieuw `Smoother` en een paar nodes toe in `Engine`, plus een PARAM-id. Het patroon herhaalt zich.
