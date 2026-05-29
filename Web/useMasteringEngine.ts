import { useCallback, useEffect, useRef, useState } from "react";

// Parameter IDs — must stay in sync with the `set_param` match in dsp/src/lib.rs.
// Treat this list as the public contract between Rust and React.
export const PARAM = {
  LOW: 0,            // dB, -12 .. +12 typical
  MID: 1,            // dB
  HIGH: 2,           // dB
  WIDTH: 3,          // 0 (mono) .. 2 (wide); 1 = identity
  OUT_GAIN_DB: 4,    // dB
  MB_ENABLED: 5,     // 0 or 1
  MB_LOW_GAIN_DB: 6, // dB
  MB_MID_GAIN_DB: 7, // dB
  MB_HIGH_GAIN_DB: 8,// dB
} as const;

export type ParamId = (typeof PARAM)[keyof typeof PARAM];

export interface MasteringEngine {
  /** AudioContext — exists once `ready` is true. Use it to decode files and create sources. */
  ctx: AudioContext | null;
  /** The worklet node. Connect a BufferSource (or any AudioNode) INTO this. */
  node: AudioWorkletNode | null;
  /** Push a parameter change. Safe to call before `ready` — calls are queued. */
  setParam: (id: ParamId, value: number) => void;
  /** True once the WASM is instantiated and routing is live. */
  ready: boolean;
  /** Non-null if loading the WASM or worklet failed. */
  error: string | null;
}

export interface UseMasteringEngineOptions {
  /** Public URL of the compiled `.wasm` (e.g. `/mastering_dsp.wasm`). */
  wasmUrl: string;
  /** Public URL of the worklet script (e.g. `/mastering-processor.js`). */
  processorUrl: string;
  /** If true (default), the engine output is connected to ctx.destination automatically. */
  autoConnect?: boolean;
}

/**
 * useMasteringEngine — wires up the Rust/WASM DSP engine inside an AudioWorklet
 * and returns a node you can connect your sources to.
 *
 *   const { ctx, node, setParam, ready } = useMasteringEngine({
 *     wasmUrl: "/mastering_dsp.wasm",
 *     processorUrl: "/mastering-processor.js",
 *   });
 *   // when the user picks a file:
 *   const buffer = await ctx!.decodeAudioData(arrayBuf);
 *   const source = ctx!.createBufferSource();
 *   source.buffer = buffer;
 *   source.connect(node!);   // <-- into the engine
 *   source.start();
 *   // when a slider moves:
 *   setParam(PARAM.LOW, +3);
 */
export function useMasteringEngine(opts: UseMasteringEngineOptions): MasteringEngine {
  const { wasmUrl, processorUrl, autoConnect = true } = opts;

  const ctxRef = useRef<AudioContext | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  // Buffer parameter changes that arrive before the worklet finishes loading.
  // Without this the first slider tweak after mount would silently miss.
  const pendingRef = useRef<Array<{ id: ParamId; value: number }>>([]);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const ctx = new AudioContext();
        ctxRef.current = ctx;

        // 1. Load the worklet module. addModule resolves once the script is
        //    evaluated inside AudioWorkletGlobalScope.
        await ctx.audioWorklet.addModule(processorUrl);
        if (cancelled) return;

        // 2. Fetch the WASM bytes on the main thread (the worklet can't fetch).
        const wasmBytes = await (await fetch(wasmUrl)).arrayBuffer();
        if (cancelled) return;

        // 3. Create the node. Force stereo output so output[1] always exists
        //    even when the upstream source is mono.
        const node = new AudioWorkletNode(ctx, "mastering-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2],
        });

        // 4. Handshake: hand the WASM to the worklet, wait for `ready`.
        const readyP = new Promise<void>((resolve, reject) => {
          node.port.onmessage = (e) => {
            if (e.data?.type === "ready") resolve();
            else if (e.data?.type === "error") reject(new Error(e.data.error));
          };
        });
        // Transfer the buffer (not copy) — saves a round-trip for a multi-MB wasm.
        node.port.postMessage({ type: "init", wasmBytes }, [wasmBytes]);
        await readyP;
        if (cancelled) return;

        // 5. Flush any params the caller wrote before we were ready.
        for (const p of pendingRef.current) {
          node.port.postMessage({ type: "param", id: p.id, value: p.value });
        }
        pendingRef.current = [];

        if (autoConnect) node.connect(ctx.destination);
        nodeRef.current = node;
        setReady(true);
      } catch (err) {
        if (!cancelled) {
          console.error("[useMasteringEngine]", err);
          setError(String(err));
        }
      }
    })();

    return () => {
      cancelled = true;
      try { nodeRef.current?.disconnect(); } catch { /* noop */ }
      void ctxRef.current?.close();
      nodeRef.current = null;
      ctxRef.current = null;
    };
  }, [wasmUrl, processorUrl, autoConnect]);

  const setParam = useCallback<MasteringEngine["setParam"]>((id, value) => {
    const node = nodeRef.current;
    if (!node) {
      pendingRef.current.push({ id, value });
      return;
    }
    node.port.postMessage({ type: "param", id, value });
  }, []);

  return {
    ctx: ctxRef.current,
    node: nodeRef.current,
    setParam,
    ready,
    error,
  };
}
