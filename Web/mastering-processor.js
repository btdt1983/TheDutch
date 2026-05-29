// mastering-processor.js — runs in AudioWorkletGlobalScope.
// Loads the Rust-compiled WASM (handed over by the main thread), then routes
// every 128-sample render quantum through the engine. No imports, because
// AudioWorkletGlobalScope doesn't support ES module imports in worklets and
// doesn't expose fetch — the main thread hands us the wasm bytes directly.

const BLOCK = 128;

class MasteringProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ready = false;
    this.exports = null;
    this.engine = 0;
    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  async onMessage(msg) {
    if (!msg) return;
    if (msg.type === 'init') {
      // Instantiate the WASM. No imports object needed — our crate uses no
      // env imports (no host calls).
      try {
        const { instance } = await WebAssembly.instantiate(msg.wasmBytes, {});
        this.exports = instance.exports;
        this.engine = this.exports.engine_new(sampleRate);

        // Float32Array views directly over the WASM memory. These let us copy
        // input/output samples without any per-frame allocation. Note: if WASM
        // memory ever grows, these views become detached — we don't grow
        // memory in this crate (no allocations after engine_new), so they
        // stay valid for the entire lifetime of the processor.
        const mem = new Float32Array(this.exports.memory.buffer);
        const inLOff  = this.exports.engine_in_l_ptr (this.engine) / 4;
        const inROff  = this.exports.engine_in_r_ptr (this.engine) / 4;
        const outLOff = this.exports.engine_out_l_ptr(this.engine) / 4;
        const outROff = this.exports.engine_out_r_ptr(this.engine) / 4;
        this.inL  = mem.subarray(inLOff,  inLOff  + BLOCK);
        this.inR  = mem.subarray(inROff,  inROff  + BLOCK);
        this.outL = mem.subarray(outLOff, outLOff + BLOCK);
        this.outR = mem.subarray(outROff, outROff + BLOCK);

        this.ready = true;
        this.port.postMessage({ type: 'ready' });
      } catch (err) {
        this.port.postMessage({ type: 'error', error: String(err) });
      }
    } else if (msg.type === 'param' && this.ready) {
      this.exports.engine_set_param(this.engine, msg.id, msg.value);
    }
  }

  process(inputs, outputs) {
    // Always return true — returning false would kill the processor and
    // permanently break the audio graph for this node.
    if (!this.ready) return true;

    const input = inputs[0];
    const output = outputs[0];

    // No source connected yet → emit silence so downstream nodes get clean
    // zeros instead of stale data.
    if (!input || input.length === 0) {
      output[0].fill(0);
      if (output[1]) output[1].fill(0);
      return true;
    }

    // Mono source → fan into both channels of the stereo engine.
    const il = input[0];
    const ir = input.length > 1 ? input[1] : il;

    this.inL.set(il);
    this.inR.set(ir);
    this.exports.engine_process(this.engine, BLOCK);
    output[0].set(this.outL);
    if (output[1]) output[1].set(this.outR);

    return true;
  }
}

registerProcessor('mastering-processor', MasteringProcessor);
