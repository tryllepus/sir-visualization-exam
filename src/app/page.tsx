'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  Flex,
  Box,
  Stack,
  Group,
  Title,
  Text,
  Button,
  NumberInput,
  Slider,
  Select,
  Paper,
  Card,
  Divider,
  Badge,
} from '@mantine/core';

// 2D canvas force-graph (no VR deps)
const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), { ssr: false });

// Resistance is the weight: Each time a node gets infected, we utilize the weight to weight the probaility of getting infected again
type Node = { id: number; state: number; resistance: number }; // 0=S, 1=I, 2=R
type Link = { source: number; target: number };
type Graph = { nodes: Node[]; links: Link[] };

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
function mulberry32(a: number) {
  return function () {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------- Graph generators ---------------- */

// Watts–Strogatz small-world (O(N*k))
function makeWS(n: number, k: number, rewireProb: number, seed = 1): Graph {
  if (k % 2 !== 0) k += 1;
  const rand = mulberry32(seed);
  // Init with restistance 0, since they havent been infected
  const nodes: Node[] = Array.from({ length: n }, (_, i) => ({ id: i, state: 0, resistance: 0 }));
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

// Barabási–Albert (scale-free)
function makeBA(n: number, m: number, seed = 1): Graph {
  const rand = mulberry32(seed);
  const nodes: Node[] = Array.from({ length: n }, (_, i) => ({ id: i, state: 0, resistance: 0  }));
  const links: Link[] = [];
  const init = Math.max(m + 1, 2);
  for (let i = 0; i < init; i++) {
    for (let j = i + 1; j < init; j++) links.push({ source: i, target: j });
  }
  const deg = Array(n).fill(0);
  for (const e of links) {
    deg[e.source as number]++;
    deg[e.target as number]++;
  }
  const bag: number[] = [];
  for (let i = 0; i < init; i++) for (let k = 0; k < deg[i]; k++) bag.push(i);

  for (let v = init; v < n; v++) {
    const targets = new Set<number>();
    while (targets.size < Math.min(m, v)) {
      const u = bag.length ? bag[Math.floor(rand() * bag.length)] : Math.floor(rand() * v);
      if (u !== v) targets.add(u);
    }
    for (const u of targets) {
      links.push({ source: v, target: u });
      deg[v]++; deg[u]++;
      bag.push(v);
      bag.push(u);
    }
  }
  return { nodes, links };
}

/* ---------------- SIR simulation ---------------- */

function simulateSIR(
  graph: Graph,
  steps: number,
  beta: number,
  gamma: number,
  delta: number, // not used now if immunity is permanent
  initialInfected: number,
  seed = 2,
  gain = 0.25 // resistance gained each recovery (cap at 1)
) {
  const n = graph.nodes.length;
  const rand = mulberry32(seed);

  // build adjacency
  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const e of graph.links) {
    const u = e.source as number, v = e.target as number;
    adj[u].push(v); adj[v].push(u);
  }

  // states: 0=S, 1=I, 2=R (R means fully immune: resistance === 1)
  const states = new Uint8Array(n);
  const resistance = new Float32Array(n); // per-node resistance in [0,1]

  // seed infections
  const picked = new Set<number>();
  const target = Math.min(initialInfected, n);
  while (picked.size < target) picked.add(Math.floor(rand() * n));
  for (const j of picked) states[j] = 1;

  const timeline: Uint8Array[] = [states.slice()];

  // per-step probabilities
  const pRec = 1 - Math.exp(-gamma);
  const pEdgeInf = 1 - Math.exp(-beta);

  const scratch = new Uint8Array(n);

  for (let t = 1; t <= steps; t++) {
    scratch.set(states);

    for (let u = 0; u < n; u++) {
      const s = states[u];

      if (s === 0) {
        // S → I (scaled by lack of resistance)
        let m = 0;
        const neigh = adj[u];
        for (let k = 0; k < neigh.length; k++) if (states[neigh[k]] === 1) m++;
        if (m > 0) {
          const baseInf = 1 - Math.pow(1 - pEdgeInf, m);
          const scaledInf = baseInf * (1 - resistance[u]); // resistance reduces infection risk
          if (rand() < scaledInf) scratch[u] = 1;
        }

      } else if (s === 1) {
        // I → (R if fully immune, else S) with resistance gain
        if (rand() < pRec) {
          const newRes = Math.min(1, resistance[u] + gain);
          resistance[u] = newRes;
          if (newRes >= 1) {
            scratch[u] = 2; // fully immune forever
          } else {
            scratch[u] = 0; // back to S with partial resistance
          }
        }

      } else if (s === 2) {
        // R = fully immune forever ⇒ nothing changes
        scratch[u] = 2;
      }
    }

    states.set(scratch);
    timeline.push(states.slice());
  }

  return timeline;
}



function seriesFromTimeline(timeline: Uint8Array[]) {
  if (!timeline || !timeline.length) return null;
  const n = timeline[0].length;
  const S: number[] = [], I: number[] = [], R: number[] = [];
  for (const arr of timeline) {
    let s = 0, i = 0, r = 0;
    for (let k = 0; k < arr.length; k++) {
      const v = arr[k];
      if (v === 0) s++; else if (v === 1) i++; else r++;
    }
    S.push(s); I.push(i); R.push(r);
  }
  const Sf = S.map(x => x / n), If = I.map(x => x / n), Rf = R.map(x => x / n);
  return { S, I, R, Sf, If, Rf, N: n };
}
function linePath(data: number[], width: number, height: number) {
  if (!data.length) return '';
  const maxX = data.length - 1;
  const sx = (i: number) => (i / maxX) * (width - 2) + 1;
  const sy = (y: number) => (1 - y) * (height - 2) + 1;
  let d = `M ${sx(0)} ${sy(data[0])}`;
  for (let i = 1; i < data.length; i++) d += ` L ${sx(i)} ${sy(data[i])}`;
  return d;
}

/* ---------------- Page ---------------- */

export default function Page() {
  // Controls
  const [population, setPopulation] = useState(100);
  const [graphType, setGraphType] = useState<'ws' | 'ba' | 'cluster'>('ws');
  // Cluster params
  const [numClusters, setNumClusters] = useState(4);
  const [intraK, setIntraK] = useState(10);
  const [interP, setInterP] = useState(0.01);
// Clustered graph: n nodes split into c clusters, each cluster is a WS, sparse random inter-cluster links
function makeClustered(n: number, c: number, k: number, interProb: number, seed = 1): Graph {
  const rand = mulberry32(seed);
  const nodes: Node[] = Array.from({ length: n }, (_, i) => ({ id: i, state: 0, resistance: 0 }));
  const links: Link[] = [];
  const clusterSize = Math.floor(n / c);
  const clusters: number[][] = [];
  let idx = 0;
  for (let ci = 0; ci < c; ci++) {
    const size = ci === c - 1 ? n - idx : clusterSize;
    const cluster = [];
    for (let j = 0; j < size; j++) cluster.push(idx++);
    clusters.push(cluster);
    // Intra-cluster WS
    const ws = makeWS(cluster.length, k, 0.05, seed + ci + 1);
    // Remap node ids
    for (const l of ws.links) {
      links.push({ source: cluster[l.source as number], target: cluster[l.target as number] });
    }
  }
  // Inter-cluster random links
  for (let i = 0; i < c; i++) {
    for (let j = i + 1; j < c; j++) {
      for (const u of clusters[i]) {
        for (const v of clusters[j]) {
          if (rand() < interProb) links.push({ source: u, target: v });
        }
      }
    }
  }
  return { nodes, links };
}

  // WS params
  const [k, setK] = useState(12);
  const [rewire, setRewire] = useState(0.05);

  // BA params
  const [mBA, setMBA] = useState(3);

  // SIR params
  const [beta, setBeta] = useState(0.02);
  const [gamma, setGamma] = useState(0.02);
  const [delta, setDelta] = useState(0.05);
  const [steps, setSteps] = useState(300);
  const [initInf, setInitInf] = useState(10);
  // Hvis brugeren ikke har valgt seed, brug random seed (baseret på tid)
  const [simSeed, setSimSeed] = useState(0); // 0 = random hver gang

  // State
  const [graph, setGraph] = useState<Graph | null>(null);
  const [timeline, setTimeline] = useState<Uint8Array[] | null>(null);
  const [t, setT] = useState(0);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const playRef = useRef<number | null>(null);

  const series = useMemo(() => (timeline ? seriesFromTimeline(timeline) : null), [timeline]);

  const simInfo = useMemo(() => {
    if (!timeline) return null;
    const states = timeline[t];
    let S = 0, I = 0, R = 0;
    for (let i = 0; i < states.length; i++) { const v = states[i]; if (v === 0) S++; else if (v === 1) I++; else R++; }
    return { S, I, R, N: states.length };
  }, [timeline, t]);

  const handleSimulate = async () => {
    setBusy(true);
    await new Promise(r => setTimeout(r, 0));
    const n = clamp(population, 10, 6000);
  // Hvis seed er 0 eller falsy, brug random seed VED HVER SIMULERING
  const usedSeed = simSeed ? simSeed : Math.floor(Math.random() * 1e9) + 1;
    let gg: Graph;
    if (graphType === 'ws') {
      gg = makeWS(n, clamp(k, 2, 200), clamp(rewire, 0, 1), usedSeed);
    } else if (graphType === 'ba') {
      gg = makeBA(n, clamp(mBA, 1, Math.max(1, Math.floor(n / 5))), usedSeed);
    } else {
      gg = makeClustered(
        n,
        clamp(numClusters, 2, Math.max(2, Math.floor(n / 10))),
        clamp(intraK, 2, 100),
        clamp(interP, 0, 1),
        usedSeed
      );
    }
    const tl = simulateSIR(
      gg,
      clamp(steps, 1, 3000),
      beta,
      gamma,
      delta,
      clamp(initInf, 1, n),
      usedSeed
    );
    setGraph(gg);
    setTimeline(tl);
    setT(0);
    setBusy(false);
    setPlaying(false);
  };

  useEffect(() => {
    if (!playing) {
      if (playRef.current) cancelAnimationFrame(playRef.current);
      playRef.current = null;
      return;
    }
    const step = () => {
      setT(prev => {
        if (!timeline) return 0;
        if (prev < timeline.length - 1) return prev + 1;
        return prev;
      });
      playRef.current = requestAnimationFrame(step);
    };
    playRef.current = requestAnimationFrame(step);
    return () => { if (playRef.current) cancelAnimationFrame(playRef.current); };
  }, [playing, timeline]);

  const nodeCanvasObject = (node: any, ctx: CanvasRenderingContext2D) => {
    const idx = node.id as number;  
    const s = timeline ? timeline[t][idx] : 0;
    const color = s === 0 ? '#60a5fa' : s === 1 ? '#f87171' : '#34d399';
    const r = 2.6;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
    ctx.fill();
  };

  const SnapshotBars = () => {
    if (!simInfo) return null;
    const { S, I, R, N } = simInfo;
    const w = 240, h = 84, pad = 10;
    const vals = [S / N, I / N, R / N];
    const colors = ['#60a5fa', '#f87171', '#34d399'];
    const bw = (w - pad * 4) / 3;
    return (
      <svg width={w} height={h}>
        {vals.map((v, i) => {
          const x = pad + i * (bw + pad);
          const bh = Math.max(1, v * (h - pad * 2));
          return <rect key={i} x={x} y={h - pad - bh} width={bw} height={bh} fill={colors[i]} rx={6} />;
        })}
        <text x={pad} y={16} fill="#94a3b8" fontSize={12}>{`t = ${t}`}</text>
      </svg>
    );
  };

  const LinesChart = () => {
    if (!series) return null;
    const { Sf, If, Rf } = series;
    const W = 680, H = 200;
    const dS = linePath(Sf, W, H);
    const dI = linePath(If, W, H);
    const dR = linePath(Rf, W, H);
    const cursorX = timeline ? (t / (timeline.length - 1)) * (W - 2) + 1 : 1;
    return (
      <svg width={W} height={H}>
        <rect x={0} y={0} width={W} height={H} rx={10} fill="#0b1220" />
        <line x1={1} y1={H - 1} x2={W - 1} y2={H - 1} stroke="#1f2937" />
        <line x1={1} y1={1} x2={1} y2={H - 1} stroke="#1f2937" />
        <path d={dS} fill="none" stroke="#60a5fa" strokeWidth={3} />
        <path d={dI} fill="none" stroke="#f87171" strokeWidth={3} />
        <path d={dR} fill="none" stroke="#34d399" strokeWidth={3} />
        <line x1={cursorX} y1={0} x2={cursorX} y2={H} stroke="#94a3b8" strokeDasharray="4 3" />
      </svg>
    );
  };

  return (
      <Flex direction="column" gap="md" p="md" style={{ minHeight: '100vh', background: '#0b0f1a' }}>
        <Group justify="space-between">
          <Title order={2} style={{color: 'white'}}>SIR on a Network (interactive)</Title>
          <Group gap="xs">
            <Badge color="blue" size="lg">S</Badge>
            <Badge color="red" size="lg">I</Badge>
            <Badge color="teal" size="lg">R</Badge>
          </Group>
        </Group>

        {/* Two-column: left sidebar fixed, right flexible */}
        <Flex gap="md" align="stretch">
          {/* Sidebar */}
          <Box w={360}>
            <Paper style={{padding: 40}}>
              <Stack gap="sm">
                <Title order={4}>Controls</Title>

                <NumberInput
                  label="Population"
                  min={10}
                  max={6000}
                  value={population}
                  onChange={(v) => setPopulation(Number(v) || 0)}
                />

                <Select
                  label="Graph type"
                  value={graphType}
                  onChange={(v) => setGraphType((v as 'ws' | 'ba' | 'cluster') ?? 'ws')}
                  data={[
                    { value: 'ws', label: 'Topology 1 (WS)' },
                    { value: 'ba', label: 'Topology 2 (BA)' },
                    { value: 'cluster', label: 'Topology 3 (Clusters)' },
                  ]}
                />
                {graphType === 'cluster' && (
                  <>
                    <NumberInput
                      label="Amount of clusters"
                      min={2}
                      max={Math.max(2, Math.floor(population / 10))}
                      value={numClusters}
                      onChange={(v) => setNumClusters(Number(v) || 0)}
                    />
                    <NumberInput
                      label="Intra cluster k (ws average)"
                      min={2}
                      max={100}
                      step={2}
                      value={intraK}
                      onChange={(v) => setIntraK(Number(v) || 0)}
                    />
                    <NumberInput
                      label="Intra cluster p"
                      min={0}
                      max={1}
                      step={0.001}
                      value={interP}
                      onChange={(v) => setInterP(Number(v) || 0)}
                    />
                  </>
                )}

                {graphType === 'ws' && (
                  <>
                    <NumberInput
                      label="k (avg degree ≈ k)"
                      min={2}
                      max={200}
                      step={2}
                      value={k}
                      onChange={(v) => setK(Number(v) || 0)}
                    />
                    <NumberInput
                      label="rewire p"
                      min={0}
                      max={1}
                      step={0.01}
                      value={rewire}
                      onChange={(v) => setRewire(Number(v) || 0)}
                    />
                  </>
                )}

                {graphType === 'ba' && (
                  <NumberInput
                    label="m (new edges per node)"
                    min={1}
                    max={Math.max(1, Math.floor(population / 5))}
                    value={mBA}
                    onChange={(v) => setMBA(Number(v) || 0)}
                  />
                )}

                <Divider label="SIR parameters" labelPosition="center" />

                <NumberInput
                  label="Seed (random, blank = ny hver gang)"
                  min={0}
                  max={999999}
                  value={simSeed}
                  onChange={(v) => setSimSeed(Number(v) || 0)}
                  placeholder="(tom = random)"
                />
                <NumberInput
                  label="β (infection)"
                  min={0}
                  max={2}
                  step={0.01}
                  value={beta}
                  onChange={(v) => setBeta(Number(v) || 0)}
                />
                <NumberInput
                  label="γ (recovery)"
                  min={0}
                  max={2}
                  step={0.01}
                  value={gamma}
                  onChange={(v) => setGamma(Number(v) || 0)}
                />
                <NumberInput
                  label="Delta (re-infection)"
                  min={0}
                  max={2}
                  step={0.01}
                  value={delta}
                  onChange={(v) => setDelta(Number(v) || 0)}
                />
                <NumberInput
                  label="initial infected"
                  min={1}
                  max={1000}
                  value={initInf}
                  onChange={(v) => setInitInf(Number(v) || 0)}
                />
                <NumberInput
                  label="steps"
                  min={10}
                  max={3000}
                  value={steps}
                  onChange={(v) => setSteps(Number(v) || 0)}
                />

                <Group grow mt="xs">
                  <Button onClick={handleSimulate} loading={busy}>
                    {busy ? 'Simulating…' : 'Simulate'}
                  </Button>
                  <Button variant="outline" onClick={() => setPlaying((p) => !p)} disabled={!timeline}>
                    {playing ? 'Pause' : 'Play'}
                  </Button>
                </Group>

                <Divider my="xs" />
                <Text size="sm" c="dimmed">
                  Scrub time
                </Text>
                <Slider
                  min={0}
                  max={(timeline?.length ?? 1) - 1}
                  value={t}
                  onChange={setT}
                  disabled={!timeline}
                  marks={
                    timeline
                      ? [
                          { value: 0, label: '0' },
                          { value: timeline.length - 1, label: String(timeline.length - 1) },
                        ]
                      : [{ value: 0, label: '0' }]
                  }
                />


              </Stack>
            </Paper>
          </Box>

          {/* Main */}
          <Flex direction="column" gap="md" style={{ flex: 1, minWidth: 0 }}>
            <Group grow>
              <Card withBorder>
                <Text size="sm" c="dimmed">
                  Susceptible
                </Text>
                <Title order={3} c="blue">
                  {simInfo?.S ?? 0}
                </Title>
              </Card>
              <Card withBorder>
                <Text size="sm" c="dimmed">
                  Infected
                </Text>
                <Title order={3} c="red">
                  {simInfo?.I ?? 0}
                </Title>
              </Card>
              <Card withBorder>
                <Text size="sm" c="dimmed">
                  Recovered
                </Text>
                <Title order={3} c="teal">
                  {simInfo?.R ?? 0}
                </Title>
              </Card>
              <Card withBorder>
                <Text size="sm" c="dimmed">
                  Time step
                </Text>
                <Title order={3}>{t}</Title>
              </Card>
            </Group>

            <Paper withBorder p="md">
              <Group justify="space-between" mb="xs">
                <Title order={4}>S / I / R (fractions)</Title>
                <Group gap="xs">
                  <Badge color="blue">S</Badge>
                  <Badge color="red">I</Badge>
                  <Badge color="teal">R</Badge>
                </Group>
              </Group>
              <Box style={{ overflowX: 'auto' }}>
                <LinesChart />
              </Box>
            </Paper>
            <Paper withBorder p="sm">
              <Text size="xs" c="dimmed" mb={6}>
                Snapshot (fractions)
              </Text>
              <SnapshotBars />
              {simInfo && (
                <Group gap="xs" mt="xs">
                  <Badge color="blue" variant="light">
                    S: {simInfo.S}
                  </Badge>
                  <Badge color="red" variant="light">
                    I: {simInfo.I}
                  </Badge>
                  <Badge color="teal" variant="light">
                    R: {simInfo.R}
                  </Badge>
                  <Text size="xs" c="dimmed" ml="auto">
                    N={simInfo.N}
                  </Text>
                </Group>
              )}
            </Paper>
            <Paper withBorder style={{ height: '64vh', overflow: 'hidden' }}>
              {graph ? (
                <ForceGraph2D
                  graphData={graph}
                  cooldownTicks={50}
                  nodeRelSize={1}
                  linkColor={() => 'rgba(156,163,175,0.25)'}
                  nodeCanvasObject={nodeCanvasObject}
                />
              ) : (
                <Group p="lg">
                  <Text c="dimmed">Click <b>Simulate</b> to generate a graph and run SIR.</Text>
                </Group>
              )}
            </Paper>
          </Flex>
        </Flex>
      </Flex>
  );
}
