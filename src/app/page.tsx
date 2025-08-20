'use client';

import { useMemo, useState } from 'react';
import dynamic from 'next/dynamic';

// Use the 2D canvas variant to avoid VR/AFRAME deps
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

type Node = { id: number; state: number }; // 0=S,1=I,2=R
type Link = { source: number; target: number };
type Graph = { nodes: Node[]; links: Link[] };

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// Simple seeded RNG
function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Watts–Strogatz small-world (O(N*k))
function makeWS(n: number, k: number, rewireProb: number, seed = 1): Graph {
  if (k % 2 !== 0) k += 1;
  const rand = mulberry32(seed);
  const nodes: Node[] = Array.from({ length: n }, (_, i) => ({ id: i, state: 0 }));
  const links: Link[] = [];
  // ring lattice
  for (let i = 0; i < n; i++) {
    for (let d = 1; d <= k / 2; d++) {
      const j = (i + d) % n;
      links.push({ source: i, target: j });
    }
  }
  // rewire
  for (let idx = 0; idx < links.length; idx++) {
    if (rand() < rewireProb) {
      const i = links[idx].source as number;
      let j = Math.floor(rand() * n);
      // avoid self-loop & duplicate
      let tries = 0;
      while ((j === i || hasEdge(links, i, j)) && tries++ < 50) j = Math.floor(rand() * n);
      links[idx].target = j;
    }
  }
  return { nodes, links };

  function hasEdge(L: Link[], a: number, b: number) {
    for (const e of L) {
      const s = e.source as number, t = e.target as number;
      if ((s === a && t === b) || (s === b && t === a)) return true;
    }
    return false;
  }
}

// ---------- Gillespie SSA implementation ----------

type SimResult = {
  times: number[];          // t at each event (times[0] = 0)
  timeline: Uint8Array[];   // state snapshot after each event (timeline[0] = initial)
};

function buildAdjacency(graph: Graph): number[][] {
  const n = graph.nodes.length;
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const e of graph.links) {
    const u = e.source as number, v = e.target as number;
    adj[u].push(v); adj[v].push(u);
  }
  return adj;
}

function gillespieSIR(
  graph: Graph,
  beta: number,      // infection rate per S–I edge
  gamma: number,     // recovery rate per I node
  initialInfected: number,
  seed = 2,
  maxEvents = 20000, // cap on number of events to simulate
  tMax = Infinity    // optional time horizon
): SimResult {
  const n = graph.nodes.length;
  const rng = mulberry32(seed);
  const U = () => Math.max(1e-12, Math.min(1 - 1e-12, rng())); // protect logs

  const adj = buildAdjacency(graph);

  // states: 0=S, 1=I, 2=R
  const states = new Uint8Array(n);
  // seed infections
  const picked = new Set<number>();
  const target = Math.min(initialInfected, n);
  while (picked.size < target) picked.add(Math.floor(rng() * n));
  for (const j of picked) states[j] = 1;

  const timeline: Uint8Array[] = [states.slice()];
  const times: number[] = [0];
  let t = 0;

  // Event loop
  for (let ev = 0; ev < maxEvents; ev++) {
    // Build event list with rates
    let Rtot = 0;
    // To avoid storing a big list, we’ll do one pass to get Rtot,
    // and another pass to pick the event (alias of cumulative).
    // Pass 1: total rate
    for (let u = 0; u < n; u++) {
      const s = states[u];
      if (s === 0) {
        // susceptible: beta * (# infected neighbors)
        let m = 0;
        const neigh = adj[u];
        for (let k = 0; k < neigh.length; k++) if (states[neigh[k]] === 1) m++;
        if (m > 0) Rtot += beta * m;
      } else if (s === 1) {
        Rtot += gamma; // recovery
      }
    }

    if (Rtot <= 0) break; // absorbing

    // Draw waiting time: dt ~ Exp(Rtot)
    const dt = -Math.log(U()) / Rtot;
    t += dt;
    if (t > tMax) break;

    // Draw which event fires
    let thresh = U() * Rtot;
    let cum = 0;

    // Pass 2: find event and apply it
    let fired = false;
    for (let u = 0; u < n; u++) {
      const s = states[u];
      if (s === 0) {
        let m = 0;
        const neigh = adj[u];
        for (let k = 0; k < neigh.length; k++) if (states[neigh[k]] === 1) m++;
        if (m > 0) {
          const r = beta * m;
          if (cum + r >= thresh) {
            // infection event at u
            states[u] = 1;
            fired = true;
            break;
          }
          cum += r;
        }
      } else if (s === 1) {
        const r = gamma;
        if (cum + r >= thresh) {
          // recovery event at u
          states[u] = 2;
          fired = true;
          break;
        }
        cum += r;
      }
    }

    if (!fired) {
      // Numerical fall-through (shouldn’t happen); stop to be safe
      break;
    }

    // Record snapshot
    timeline.push(states.slice());
    times.push(t);
  }

  return { times, timeline };
}

// --------------------------------------------------

export default function Page() {
  // UI state
  const [population, setPopulation] = useState(500);
  const [k, setK] = useState(6);
  const [rewire, setRewire] = useState(0.05);
  const [beta, setBeta] = useState(0.15);
  const [gamma, setGamma] = useState(0.3);
  const [maxEvents, setMaxEvents] = useState(5000); // formerly "steps"
  const [initInf, setInitInf] = useState(5);

  const [graph, setGraph] = useState<Graph | null>(null);
  const [timeline, setTimeline] = useState<Uint8Array[] | null>(null);
  const [times, setTimes] = useState<number[] | null>(null);
  const [idx, setIdx] = useState(0); // event index
  const [busy, setBusy] = useState(false);

  const simInfo = useMemo(() => {
    if (!timeline) return null;
    const states = timeline[idx];
    let S = 0, I = 0, R = 0;
    for (let i = 0; i < states.length; i++) {
      if (states[i] === 0) S++;
      else if (states[i] === 1) I++;
      else R++;
    }
    return { S, I, R };
  }, [timeline, idx]);

  const handleSimulate = async () => {
    setBusy(true);
    await new Promise((r) => setTimeout(r, 0)); // yield to paint
    const n = clamp(population, 10, 6000);
    const gg = makeWS(n, clamp(k, 2, 40), clamp(rewire, 0, 1), 1);
    const { times, timeline } = gillespieSIR(
      gg,
      beta,
      gamma,
      clamp(initInf, 1, Math.max(1, Math.floor(n * 0.1))),
      2,
      clamp(maxEvents, 1, 200000),
      Infinity
    );
    setGraph(gg);
    setTimeline(timeline);
    setTimes(times);
    setIdx(0);
    setBusy(false);
  };

  // color by state at current event
  const nodeCanvasObject = (node: any, ctx: CanvasRenderingContext2D) => {
    const i = node.id as number;
    const s = timeline ? timeline[idx][i] : 0;
    const color = s === 0 ? '#1f77b4' : s === 1 ? '#d62728' : '#2ca02c'; // S/I/R
    const r = 2.2;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
    ctx.fill();
  };

  return (
    <div className="min-h-screen p-4 flex flex-col gap-4">
      <h1 style={{ fontSize: '1.4rem', fontWeight: 600 }}>SIR on a Network (Gillespie SSA)</h1>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(6, minmax(0,1fr))',
          gap: '8px',
          alignItems: 'end',
        }}
      >
        <label>
          Population
          <input
            type="number"
            min={10}
            max={6000}
            value={population}
            onChange={(e) => setPopulation(parseInt(e.target.value || '0'))}
          />
        </label>
        <label>
          k (avg degree≈k)
          <input
            type="number"
            min={2}
            max={40}
            step={2}
            value={k}
            onChange={(e) => setK(parseInt(e.target.value || '0'))}
          />
        </label>
        <label>
          rewire p
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={rewire}
            onChange={(e) => setRewire(parseFloat(e.target.value || '0'))}
          />
        </label>
        <label>
          β (infection rate)
          <input
            type="number"
            min={0}
            max={2}
            step={0.01}
            value={beta}
            onChange={(e) => setBeta(parseFloat(e.target.value || '0'))}
          />
        </label>
        <label>
          γ (recovery rate)
          <input
            type="number"
            min={0}
            max={2}
            step={0.01}
            value={gamma}
            onChange={(e) => setGamma(parseFloat(e.target.value || '0'))}
          />
        </label>
        <label>
          initial infected
          <input
            type="number"
            min={1}
            max={1000}
            value={initInf}
            onChange={(e) => setInitInf(parseInt(e.target.value || '0'))}
          />
        </label>

        <label>
          max events
          <input
            type="number"
            min={10}
            max={200000}
            value={maxEvents}
            onChange={(e) => setMaxEvents(parseInt(e.target.value || '0'))}
          />
        </label>
        <button onClick={handleSimulate} disabled={busy} style={{ gridColumn: 'span 2' }}>
          {busy ? 'Simulating…' : 'Simulate'}
        </button>

        <div style={{ gridColumn: 'span 3', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <input
            type="range"
            min={0}
            max={(timeline?.length ?? 1) - 1}
            value={idx}
            onChange={(e) => setIdx(parseInt(e.target.value || '0'))}
            style={{ width: '100%' }}
            disabled={!timeline}
          />
          <span>
            event: {idx}/{(timeline?.length ?? 1) - 1}
            {times && times.length > 0 ? ` • time t ≈ ${times[idx].toFixed(3)}` : ''}
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
        <div>
          Legend: <span style={{ color: '#1f77b4' }}>● S</span>{' '}
          <span style={{ color: '#d62728' }}>● I</span>{' '}
          <span style={{ color: '#2ca02c' }}>● R</span>
        </div>
        {simInfo && (
          <div>
            Counts @event {idx}: S={simInfo.S} | I={simInfo.I} | R={simInfo.R}
          </div>
        )}
      </div>

      <div style={{ height: '70vh', border: '1px solid #ddd', borderRadius: 8 }}>
        {graph ? (
          <ForceGraph2D
            graphData={graph}
            cooldownTicks={50}
            nodeRelSize={1}
            nodeLabel={(n: any) => `id: ${n.id}`}
            linkColor={() => 'rgba(120,120,120,0.3)'}
            nodeCanvasObject={nodeCanvasObject}
          />
        ) : (
          <div style={{ padding: 16 }}>
            Click <b>Simulate</b> to generate a graph and run SIR (Gillespie).
          </div>
        )}
      </div>
    </div>
  );
}
