"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type { DashRun, Dashboard } from "@/lib/types";

// --------------------------------------------------------------------------- //
// shared bits
// --------------------------------------------------------------------------- //

// mid-tone categorical colours: legible on the light and the dark surface
const AGENT_COLORS: Record<string, string> = {
  "mle-star": "#3f7fd9", mlevolve: "#e0931b", aide: "#d1495b",
};
const SPARE = ["#2a9d8f", "#9b5de5", "#7a9a3a", "#c26bb0"];
const UNKNOWN = "#8a8a8a";
const NO_MODEL = "⟨unknown / custom⟩";
/** names that are real models, without the no-model bucket */
const modelNames = (u: Record<string, number>) => Object.keys(u).filter((k) => k !== NO_MODEL);
const agentName = (a: string | null) => a ?? "unknown";

function useAgentColor(agents: string[]) {
  return useMemo(() => {
    const m = new Map<string, string>();
    let k = 0;
    for (const a of agents) m.set(a, AGENT_COLORS[a] ?? (a === "unknown" ? UNKNOWN : SPARE[k++ % SPARE.length]));
    return (a: string | null) => m.get(agentName(a)) ?? UNKNOWN;
  }, [agents]);
}

const S = {
  grid: { stroke: "var(--border)", strokeWidth: 1 },
  axis: { stroke: "var(--muted)", strokeWidth: 1 },
  tick: { fill: "var(--muted)", fontSize: 10 },
  label: { fill: "var(--fg)", fontSize: 11 },
};

const pct = (v: number) => `${Math.round(v * 100)}%`;

function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function useWidth(fallback = 900) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

function Chips<T extends string>({ all, on, set, color, count }: {
  all: T[]; on: Set<T>; set: (s: Set<T>) => void; color?: (v: T) => string; count?: (v: T) => number;
}) {
  const toggle = (v: T) => {
    const n = new Set(on);
    if (n.has(v)) n.delete(v); else n.add(v);
    set(n);
  };
  return (
    <span className="dash-chips">
      {all.map((v) => (
        <button key={v} type="button" className={on.has(v) ? "on" : ""} onClick={() => toggle(v)}
                onDoubleClick={() => set(new Set([v]))} title="click to toggle, double-click for only this">
          {color && <i style={{ background: color(v) }} />}{v}
          {count && <span className="muted"> {count(v)}</span>}
        </button>
      ))}
      <button type="button" className="link-button small" onClick={() => set(new Set(all))}>all</button>
    </span>
  );
}

function Seg<T extends string>({ value, set, options }: {
  value: T; set: (v: T) => void; options: [T, string][];
}) {
  return (
    <span className="dash-seg">
      {options.map(([v, l]) => (
        <button key={v} type="button" className={v === value ? "on" : ""} onClick={() => set(v)}>{l}</button>
      ))}
    </span>
  );
}

function Panel({ title, note, controls, children, wide }: {
  title: string; note?: React.ReactNode; controls?: React.ReactNode; children: React.ReactNode; wide?: boolean;
}) {
  return (
    <section className={`dash-panel${wide ? " wide" : ""}`}>
      <div className="dash-panel-head">
        <h2>{title}</h2>
        {controls && <div className="dash-controls">{controls}</div>}
      </div>
      {note && <p className="muted small dash-note">{note}</p>}
      {children}
    </section>
  );
}

// --------------------------------------------------------------------------- //
// component usage: fold self-defined networks, pick the kind
// --------------------------------------------------------------------------- //

type Source = "model" | "ensemble" | "transformer" | "splitter" | "search" | "metric" | "libraries" | "family";
const SOURCES: [Source, string][] = [
  ["model", "models"], ["family", "model library"], ["ensemble", "ensembles"], ["transformer", "transformers"],
  ["splitter", "validation"], ["search", "HP search"], ["metric", "metrics"], ["libraries", "imports"],
];

/** scripts of ``r`` using each name of the source; a script counts once per name */
function usageOf(r: DashRun, src: Source, fold: boolean, merge = true): Record<string, number> {
  if (src === "libraries") return r.libraries;
  if (src === "metric") {
    const out: Record<string, number> = { ...r.metrics };
    for (const [k, v] of Object.entries(r.usage.metric ?? {})) out[k] = Math.max(out[k] ?? 0, v);
    return out;
  }
  if (src !== "model") return r.usage[src] ?? {};
  const key = fold && merge ? "model_folded_merged" : fold ? "model_folded" : merge ? "model_merged" : "model";
  return r.usage[key] ?? {};
}

// --------------------------------------------------------------------------- //
// heatmap: component × (dataset | agent | run), cell = share of scripts
// --------------------------------------------------------------------------- //

type Axis = "dataset" | "agent" | "run";

function Heatmap({ runs, color }: { runs: DashRun[]; color: (a: string | null) => string }) {
  const [src, setSrc] = useState<Source>("model");
  const [axis, setAxis] = useState<Axis>("dataset");
  const [fold, setFold] = useState(true);
  const [merge, setMerge] = useState(true);
  const [top, setTop] = useState(20);
  const [hover, setHover] = useState<string | null>(null);

  const cols = useMemo(() => {
    const m = new Map<string, { key: string; label: string; runs: DashRun[]; agent?: string | null }>();
    for (const r of runs) {
      const key = axis === "dataset" ? r.dataset : axis === "agent" ? agentName(r.agent) : r.id;
      const label = axis === "dataset" ? r.dataset_label : axis === "agent" ? key : r.id;
      if (!m.has(key)) m.set(key, { key, label, runs: [], agent: axis === "run" ? r.agent : undefined });
      m.get(key)!.runs.push(r);
    }
    return [...m.values()];
  }, [runs, axis]);

  const { rows, cell, total } = useMemo(() => {
    const per = new Map<string, Map<string, number>>();     // col -> name -> scripts
    const tot = new Map<string, number>();                  // name -> scripts, all columns
    const den = new Map<string, number>();
    for (const c of cols) {
      const m = new Map<string, number>();
      let n = 0;
      for (const r of c.runs) {
        n += r.n_ok;
        for (const [k, v] of Object.entries(usageOf(r, src, fold, merge))) {
          m.set(k, (m.get(k) ?? 0) + v);
          tot.set(k, (tot.get(k) ?? 0) + v);
        }
      }
      per.set(c.key, m);
      den.set(c.key, n);
    }
    // rank by mean share over columns, so a small dataset's favourite still shows
    const score = (k: string) => cols.reduce((s, c) => s + (per.get(c.key)!.get(k) ?? 0) / Math.max(den.get(c.key)!, 1), 0);
    const rows = [...tot.keys()].sort((a, b) => score(b) - score(a));
    return {
      rows,
      total: tot,
      cell: (k: string, c: string) => {
        const n = per.get(c)!.get(k) ?? 0;
        return { n, of: den.get(c)!, share: n / Math.max(den.get(c)!, 1) };
      },
    };
  }, [cols, src, fold, merge]);

  const shown = rows.slice(0, top);
  const cw = axis === "run" ? 26 : axis === "agent" ? 90 : 64;
  return (
    <Panel title="What the agents use" wide
      note={<>Share of scripts that use each component (ranked by mean share over the columns). A script counts once
        per name; what was <i>written</i>, so an unused branch counts too.{" "}
        {src === "family" && "Model library groups models by the package they come from; networks the scripts define themselves are one bucket."}</>}
      controls={<>
        <Seg value={src} set={setSrc} options={SOURCES} />
        <span className="muted small">by</span>
        <Seg value={axis} set={setAxis} options={[["dataset", "dataset"], ["agent", "agent"], ["run", "run"]]} />
        {src === "model" && <label className="small"><input type="checkbox" checked={fold}
          onChange={(e) => setFold(e.target.checked)} /> fold self-defined networks</label>}
        {src === "model" && <label className="small" title="LGBMClassifier, LGBMRegressor and lightgbm.train count as one LightGBM; different kernels (HistGradientBoosting vs LightGBM) stay apart">
          <input type="checkbox" checked={merge}
          onChange={(e) => setMerge(e.target.checked)} /> merge variants</label>}
      </>}>
      <div className="table-scroll">
        <table className="dash-heat small">
          <thead>
            <tr>
              <th />
              {cols.map((c) => (
                <th key={c.key} className={axis === "run" ? "vert" : ""} title={`${c.label}: ${c.runs.length} run(s), ${c.runs.reduce((s, r) => s + r.n_ok, 0)} scripts`}
                    style={{ width: cw, borderBottom: c.agent !== undefined ? `3px solid ${color(c.agent)}` : undefined }}>
                  {axis === "agent" && <i className="dot" style={{ background: color(c.key) }} />}
                  <span>{axis === "run" ? c.label.replace("/", " / ") : c.label}</span>
                </th>
              ))}
              <th className="num">all</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((k) => (
              <tr key={k} className={hover === k ? "hl" : ""} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
                <td className="mono clip" title={k}>{k}</td>
                {cols.map((c) => {
                  const v = cell(k, c.key);
                  return (
                    <td key={c.key} className="heat" title={`${k} · ${c.label}: ${v.n}/${v.of} scripts (${pct(v.share)})`}
                        style={{ background: v.n ? `color-mix(in srgb, var(--accent) ${Math.round(12 + v.share * 88)}%, transparent)` : undefined,
                                 color: v.share > 0.5 ? "var(--bg)" : undefined }}>
                      {v.n ? (axis === "run" ? "" : pct(v.share)) : ""}
                    </td>
                  );
                })}
                <td className="num muted">{total.get(k)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > top && (
        <p className="small"><button type="button" className="link-button" onClick={() => setTop(top + 30)}>
          show more</button> <span className="muted">({rows.length - top} more)</span></p>
      )}
    </Panel>
  );
}

// --------------------------------------------------------------------------- //
// change CDF: every run's ratio of changed lines, or pooled per agent
// --------------------------------------------------------------------------- //

function CdfPanel({ runs, color, agents }: { runs: DashRun[]; color: (a: string | null) => string; agents: string[] }) {
  const [mode, setMode] = useState<"agent" | "run" | "dataset">("agent");
  const [log, setLog] = useState(false);
  const [hl, setHl] = useState<string | null>(null);
  const [box, width] = useWidth();
  const W = Math.min(width, 760), PH = 240, m = { top: 10, right: 16, bottom: 34, left: 42 };
  const pw = W - m.left - m.right;

  const series = useMemo(() => {
    const ratios = (rs: DashRun[]) => rs.flatMap((r) => r.steps.map((s) => s.ratio).filter((x): x is number => x != null))
      .sort((a, b) => a - b);
    if (mode === "run") return runs.map((r) => ({ key: r.id, color: color(r.agent), v: ratios([r]), sub: agentName(r.agent) }));
    if (mode === "agent") return agents.map((a) => ({ key: a, color: color(a), v: ratios(runs.filter((r) => agentName(r.agent) === a)), sub: "" }));
    const ds = [...new Set(runs.map((r) => r.dataset))];
    return ds.map((d, i) => ({ key: d, color: `hsl(${(i * 360) / ds.length} 55% 52%)`, v: ratios(runs.filter((r) => r.dataset === d)), sub: "" }));
  }, [runs, mode, agents, color]);

  const all = series.flatMap((s) => s.v);
  if (all.length === 0) return null;
  const hi = Math.max(1, Math.ceil(quantile([...all].sort((a, b) => a - b), 0.99) * 4) / 4);
  const lo = 0.001;
  const x = (r: number) => m.left + (log
    ? (Math.log10(Math.max(r, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))
    : Math.min(r, hi) / hi) * pw;
  const y = (f: number) => m.top + PH - f * PH;
  const xt = log ? [0.001, 0.01, 0.1, 1].filter((t) => t <= hi) : Array.from({ length: Math.round(hi / 0.25) + 1 }, (_, i) => i * 0.25)
    .filter((_, i, a) => a.length <= 7 || i % 2 === 0);

  const path = (v: number[]) => {
    let d = `M${x(log ? lo : 0)},${y(0)}`;
    v.forEach((r, i) => { d += `H${x(r)}V${y((i + 1) / v.length)}`; });
    return d + `H${x(hi)}`;
  };

  return (
    <Panel title="How much each step changes"
      note={<>Empirical CDF of the ratio of changed code lines against the parent, over every parent → child step; comments and formatting are ignored, so an agent that re-emits the whole file is not counted as rewriting it.
        Left and steep: small edits; a long right tail: rewrites. ○ in the table is a step that repeats its parent.</>}
      controls={<>
        <Seg value={mode} set={setMode} options={[["agent", "per agent"], ["dataset", "per dataset"], ["run", "per run"]]} />
        <label className="small"><input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> log x</label>
      </>}>
      <div ref={box} className="dash-cdf">
        <svg width={W} height={PH + m.top + m.bottom}>
          {[0.25, 0.5, 0.75, 1].map((f) => <line key={f} x1={m.left} x2={m.left + pw} y1={y(f)} y2={y(f)} style={S.grid} />)}
          <line x1={m.left} x2={m.left + pw} y1={y(0)} y2={y(0)} style={S.axis} />
          {[0, 0.25, 0.5, 0.75, 1].map((f) => <text key={f} x={m.left - 6} y={y(f) + 3} style={S.tick} textAnchor="end">{pct(f)}</text>)}
          {xt.map((t) => <text key={t} x={x(t)} y={y(0) + 13} style={S.tick} textAnchor="middle">{t < 0.01 ? "0.1%" : pct(t)}</text>)}
          <text x={m.left + pw / 2} y={y(0) + 28} style={S.tick} textAnchor="middle">ratio of changed code lines</text>
          {series.filter((s) => s.v.length).map((s) => (
            <path key={s.key} d={path(s.v)} fill="none" stroke={s.color}
                  strokeWidth={hl === s.key ? 3 : mode === "run" ? 1.3 : 2.2}
                  opacity={hl && hl !== s.key ? 0.15 : mode === "run" ? 0.75 : 1}
                  onMouseEnter={() => setHl(s.key)} onMouseLeave={() => setHl(null)}>
              <title>{s.key}: {s.v.length} steps, median {pct(median(s.v)!)}</title>
            </path>
          ))}
        </svg>
        <table className="small dash-legend">
          <thead><tr><th /><th className="num">steps</th><th className="num">median</th><th className="num">p90</th><th className="num">○ same</th></tr></thead>
          <tbody>
            {series.map((s) => {
              const same = runs.filter((r) => mode === "run" ? r.id === s.key : mode === "agent" ? agentName(r.agent) === s.key : r.dataset === s.key)
                .flatMap((r) => r.steps).filter((st) => st.same).length;
              return (
                <tr key={s.key} className={hl === s.key ? "hl" : ""} onMouseEnter={() => setHl(s.key)} onMouseLeave={() => setHl(null)}>
                  <td className="clip" title={s.key}><i className="dot" style={{ background: s.color }} />{s.key}</td>
                  <td className="num">{s.v.length}</td>
                  <td className="num">{s.v.length ? pct(median(s.v)!) : "—"}</td>
                  <td className="num">{s.v.length ? pct(quantile(s.v, 0.9)) : "—"}</td>
                  <td className="num muted">{same}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

// --------------------------------------------------------------------------- //
// along the trajectory: normalised position (0 = first script, 1 = last)
// --------------------------------------------------------------------------- //

type Along = "ratio" | "loc" | "locrel" | "cx" | "params";
const ALONG: [Along, string][] = [
  ["ratio", "change ratio"], ["loc", "lines of code"], ["locrel", "size vs first script"],
  ["cx", "complexity"], ["params", "hyperparameters changed"],
];

function AlongPanel({ runs, color, agents }: { runs: DashRun[]; color: (a: string | null) => string; agents: string[] }) {
  const [what, setWhat] = useState<Along>("ratio");
  const [showRuns, setShowRuns] = useState(true);
  const [hl, setHl] = useState<string | null>(null);
  const BINS = 10;
  const [box, width] = useWidth();
  const W = Math.min(width, 760), PH = 220, m = { top: 10, right: 16, bottom: 34, left: 48 };
  const pw = W - m.left - m.right;

  const val = (r: DashRun, i: number): number | null => {
    const s = r.steps[i];
    if (what === "ratio") return s.ratio;
    if (what === "loc") return s.ok ? s.loc : null;
    if (what === "cx") return s.cx;
    if (what === "params") return s.params;
    const first = r.steps.find((x) => x.ok)?.loc;
    return s.ok && first ? s.loc / first : null;
  };
  const binned = (rs: DashRun[]) => {
    const b: number[][] = Array.from({ length: BINS }, () => []);
    for (const r of rs) r.steps.forEach((_, i) => {
      const v = val(r, i);
      if (v != null) b[Math.min(BINS - 1, Math.floor((i / Math.max(r.steps.length, 1)) * BINS))].push(v);
    });
    return b.map(median);
  };
  const perRun = runs.map((r) => ({ r, v: binned([r]) }));
  const perAgent = agents.map((a) => ({ a, v: binned(runs.filter((r) => agentName(r.agent) === a)) }));
  const vals = [...perAgent.flatMap((p) => p.v), ...(showRuns ? perRun.flatMap((p) => p.v) : [])].filter((v): v is number => v != null);
  if (vals.length === 0) return null;
  const sorted = [...vals].sort((a, b) => a - b);
  const top = quantile(sorted, 0.97) * 1.1 || 1;
  const x = (i: number) => m.left + ((i + 0.5) / BINS) * pw;
  const y = (v: number) => m.top + PH - (Math.min(v, top) / top) * PH;
  const line = (v: (number | null)[]) => v.map((p, i) => p == null ? "" : `${v[i - 1] == null ? "M" : "L"}${x(i)},${y(p)}`).join("");
  const fmt = (v: number) => what === "ratio" ? pct(v) : what === "locrel" ? `${v.toFixed(1)}×` : Math.round(v).toLocaleString();

  return (
    <Panel title="Along the trajectory"
      note={<>Each run&apos;s scripts in order, cut into {BINS} equal slices of the trajectory; the median per slice.
        Thick: pooled per agent; thin: each run. Does the search settle into small edits, and does the code grow?</>}
      controls={<>
        <Seg value={what} set={setWhat} options={ALONG} />
        <label className="small"><input type="checkbox" checked={showRuns} onChange={(e) => setShowRuns(e.target.checked)} /> runs</label>
      </>}>
      <div ref={box}>
        <svg width={W} height={PH + m.top + m.bottom}>
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line x1={m.left} x2={m.left + pw} y1={y(top * f)} y2={y(top * f)} style={f ? S.grid : S.axis} />
              <text x={m.left - 6} y={y(top * f) + 3} style={S.tick} textAnchor="end">{fmt(top * f)}</text>
            </g>
          ))}
          {what === "locrel" && top > 1 && <line x1={m.left} x2={m.left + pw} y1={y(1)} y2={y(1)} style={S.axis} strokeDasharray="4 3" />}
          <text x={m.left} y={y(0) + 14} style={S.tick}>start</text>
          <text x={m.left + pw} y={y(0) + 14} style={S.tick} textAnchor="end">end</text>
          <text x={m.left + pw / 2} y={y(0) + 28} style={S.tick} textAnchor="middle">position in the trajectory</text>
          {showRuns && perRun.map(({ r, v }) => (
            <path key={r.id} d={line(v)} fill="none" stroke={color(r.agent)} strokeWidth={hl === r.id ? 2.5 : 1}
                  opacity={hl ? (hl === r.id ? 1 : 0.1) : 0.35}
                  onMouseEnter={() => setHl(r.id)} onMouseLeave={() => setHl(null)}>
              <title>{r.id} ({r.steps.length} scripts)</title>
            </path>
          ))}
          {perAgent.map(({ a, v }) => (
            <g key={a} opacity={hl && hl !== a && !perRun.some((p) => p.r.id === hl) ? 0.25 : 1}>
              <path d={line(v)} fill="none" stroke={color(a)} strokeWidth={3} strokeLinejoin="round"
                    onMouseEnter={() => setHl(a)} onMouseLeave={() => setHl(null)}><title>{a}</title></path>
              {v.map((p, i) => p == null ? null : (
                <circle key={i} cx={x(i)} cy={y(p)} r={3.2} fill={color(a)} stroke="var(--surface)" strokeWidth={1.5}>
                  <title>{a}, slice {i + 1}/{BINS}: {fmt(p)}</title>
                </circle>
              ))}
            </g>
          ))}
        </svg>
      </div>
    </Panel>
  );
}

// --------------------------------------------------------------------------- //
// exploration: distinct models vs trajectory length, one dot per run
// --------------------------------------------------------------------------- //

function ExplorePanel({ runs, color }: { runs: DashRun[]; color: (a: string | null) => string }) {
  const [yk, setYk] = useState<"models" | "families" | "libs">("models");
  const [box, width] = useWidth();
  const W = Math.min(width, 520), PH = 220, m = { top: 12, right: 16, bottom: 34, left: 40 };
  const pw = W - m.left - m.right;
  const pts = runs.map((r) => ({
    r, x: r.n,
    y: yk === "models" ? modelNames(usageOf(r, "model", true)).length
      : yk === "families" ? modelNames(usageOf(r, "family", false)).length
      : Object.keys(r.libraries).length,
  }));
  const xmax = Math.max(10, ...pts.map((p) => p.x)), ymax = Math.max(4, ...pts.map((p) => p.y));
  const lx = (v: number) => Math.log10(v + 1) / Math.log10(xmax + 1);
  const x = (v: number) => m.left + lx(v) * pw;
  const y = (v: number) => m.top + PH - (v / ymax) * PH;
  const xt = [1, 3, 10, 30, 100, 300].filter((t) => t <= xmax);
  return (
    <Panel title="Breadth of the search"
      note="One dot per run: how many scripts it wrote (log scale) against how many distinct things they use. Self-defined networks count as one model."
      controls={<Seg value={yk} set={setYk} options={[["models", "models"], ["families", "model libraries"], ["libs", "imports"]]} />}>
      <div ref={box}>
        <svg width={W} height={PH + m.top + m.bottom}>
          {[0, 0.5, 1].map((f) => (
            <g key={f}>
              <line x1={m.left} x2={m.left + pw} y1={y(ymax * f)} y2={y(ymax * f)} style={f ? S.grid : S.axis} />
              <text x={m.left - 6} y={y(ymax * f) + 3} style={S.tick} textAnchor="end">{Math.round(ymax * f)}</text>
            </g>
          ))}
          {xt.map((t) => <text key={t} x={x(t)} y={y(0) + 13} style={S.tick} textAnchor="middle">{t}</text>)}
          <text x={m.left + pw / 2} y={y(0) + 28} style={S.tick} textAnchor="middle">scripts in the run</text>
          {pts.map((p) => (
            <Link key={p.r.id} href={`/runs/${p.r.dataset}/${p.r.run}`}>
              <circle cx={x(p.x)} cy={y(p.y)} r={6} fill={color(p.r.agent)} fillOpacity={0.8} stroke="var(--surface)" strokeWidth={1.5}>
                <title>{p.r.id} ({agentName(p.r.agent)}): {p.x} scripts, {p.y} distinct</title>
              </circle>
            </Link>
          ))}
        </svg>
      </div>
    </Panel>
  );
}

// --------------------------------------------------------------------------- //
// script size: every script as a dot, one row per run, grouped by dataset
// --------------------------------------------------------------------------- //

function SizePanel({ runs, color }: { runs: DashRun[]; color: (a: string | null) => string }) {
  const [metric, setMetric] = useState<"loc" | "cx">("loc");
  const [box, width] = useWidth();
  const RH = 16, LW = 230, m = { top: 6, right: 16, bottom: 26 };
  const pw = Math.max(width - LW - m.right, 100);
  const groups = useMemo(() => {
    const g = new Map<string, DashRun[]>();
    for (const r of runs) g.set(r.dataset_label, [...(g.get(r.dataset_label) ?? []), r]);
    return [...g.entries()];
  }, [runs]);
  const v = (s: DashRun["steps"][number]) => metric === "loc" ? (s.ok ? s.loc : null) : s.cx;
  const all = runs.flatMap((r) => r.steps.map(v)).filter((x): x is number => x != null);
  if (all.length === 0) return null;
  const max = Math.max(...all);
  const x = (n: number) => LW + Math.sqrt(n / max) * pw;
  const ticks = [0, 0.05, 0.2, 0.5, 1].map((f) => Math.round(max * f));
  // a header row per dataset, then one row per run
  let row = 0;
  const placed = groups.map(([ds, rs]) => {
    const head = m.top + row * RH + RH * 0.9;
    const out = { ds, head, rs: rs.map((r, i) => ({ r, y: m.top + (row + i + 1.6) * RH })) };
    row += rs.length + 1;
    return out;
  });
  const H = m.top + (row + 1) * RH + m.bottom;
  return (
    <Panel title="Script size" wide
      note={<>Every script as a tick, one row per run (√ scale). The bar is the run&apos;s middle half (p25–p75), ◆ the best-scoring script.</>}
      controls={<Seg value={metric} set={setMetric} options={[["loc", "lines of code"], ["cx", "cyclomatic complexity"]]} />}>
      <div ref={box}>
        <svg width={width} height={H}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={m.top} y2={H - m.bottom} style={S.grid} />
              <text x={x(t)} y={H - m.bottom + 14} style={S.tick} textAnchor="middle">{t.toLocaleString()}</text>
            </g>
          ))}
          {placed.map((g) => (
            <g key={g.ds}>
              <text x={0} y={g.head} style={{ ...S.label, fontWeight: 600 }}>{g.ds}</text>
              {g.rs.map(({ r, y }) => {
                const vs = r.steps.map(v).filter((n): n is number => n != null).sort((a, b) => a - b);
                const best = r.steps.find((s) => s.name === r.best);
                const bv = best ? v(best) : null;
                return (
                  <g key={r.id}>
                    <Link href={`/runs/${r.dataset}/${r.run}`}>
                      <text x={10} y={y + 3} style={S.tick}><tspan fill={color(r.agent)}>●</tspan> {r.run} <tspan opacity={0.7}>({r.n})</tspan></text>
                    </Link>
                    {vs.length > 1 && <rect x={x(quantile(vs, 0.25))} y={y - 5} width={Math.max(1, x(quantile(vs, 0.75)) - x(quantile(vs, 0.25)))}
                                            height={10} rx={3} fill={color(r.agent)} opacity={0.18} />}
                    {r.steps.map((s) => {
                      const n = v(s);
                      const own = metric === "loc" && s.own_loc != null ? ` (${s.own_loc} in the script, rest inlined)` : "";
                      return n == null ? null : <line key={s.name} x1={x(n)} x2={x(n)} y1={y - 5} y2={y + 5} stroke={color(r.agent)}
                                                       strokeWidth={1.2} opacity={0.55}><title>{s.name}: {n}{own}</title></line>;
                    })}
                    {bv != null && <path d={`M${x(bv)},${y - 6}l5,6l-5,6l-5,-6z`} fill="var(--fg)"><title>best: {r.best} ({bv})</title></path>}
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
    </Panel>
  );
}

// --------------------------------------------------------------------------- //
// per run table
// --------------------------------------------------------------------------- //

function RunTable({ runs, color }: { runs: DashRun[]; color: (a: string | null) => string }) {
  const [sort, setSort] = useState<{ k: string; d: 1 | -1 }>({ k: "dataset", d: 1 });
  const rows = runs.map((r) => {
    const ok = r.steps.filter((s) => s.ok);
    const ratios = r.steps.map((s) => s.ratio).filter((x): x is number => x != null);
    const diffs = r.steps.filter((s) => s.same != null);
    const models = usageOf(r, "model", true);
    const topModel = Object.entries(models).sort((a, b) => b[1] - a[1])[0];
    return {
      r, dataset: r.dataset_label, run: r.run, agent: agentName(r.agent), n: r.n,
      loc: median(ok.map((s) => s.loc)), first: ok[0]?.loc ?? null, last: ok.at(-1)?.loc ?? null,
      ratio: median(ratios), same: diffs.length ? diffs.filter((s) => s.same).length / diffs.length : null,
      models: modelNames(models).length, top: topModel, gpu: r.n_ok ? r.gpu / r.n_ok : null, pip: r.pip,
      failed: r.n - r.n_ok,
    };
  });
  type Row = (typeof rows)[number];
  const key = (row: Row): string | number => {
    const v = row[sort.k as keyof Row];
    return typeof v === "number" || typeof v === "string" ? v : v == null ? -Infinity : 0;
  };
  rows.sort((a, b) => {
    const ka = key(a), kb = key(b);
    return (ka < kb ? -1 : ka > kb ? 1 : 0) * sort.d || a.run.localeCompare(b.run);
  });
  const Th = ({ k, children, num, title }: { k: string; children: React.ReactNode; num?: boolean; title?: string }) => (
    <th className={`${num ? "num " : ""}sortable`} title={title}
        onClick={() => setSort({ k, d: sort.k === k ? (-sort.d as 1 | -1) : num ? -1 : 1 })}>
      {children}{sort.k === k ? (sort.d === 1 ? " ▲" : " ▼") : ""}
    </th>
  );
  return (
    <Panel title="Runs" wide>
      <div className="table-scroll">
        <table className="small">
          <thead>
            <tr>
              <Th k="dataset">dataset</Th><Th k="run">run</Th><Th k="agent">agent</Th>
              <Th k="n" num>scripts</Th><Th k="loc" num title="median lines of code">loc</Th>
              <Th k="first" num title="lines of code of the first and the last script">first → last</Th>
              <Th k="ratio" num title="median ratio of changed code lines against the parent">Δ ratio</Th>
              <Th k="same" num title="steps whose code repeats the parent's, ignoring comments and formatting">same code</Th>
              <Th k="models" num title="distinct models (self-defined networks folded)">models</Th>
              <th>most used model</th>
              <Th k="gpu" num title="scripts using a GPU">GPU</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((x) => (
              <tr key={x.r.id}>
                <td className="muted">{x.dataset}</td>
                <td><Link href={`/runs/${x.r.dataset}/${x.r.run}`}>{x.run}</Link></td>
                <td><i className="dot" style={{ background: color(x.r.agent) }} />{x.agent}</td>
                <td className="num">{x.n}{x.failed > 0 && <span className="partial" title="do not parse"> ({x.failed}✗)</span>}</td>
                <td className="num">{x.loc != null ? Math.round(x.loc) : "—"}</td>
                <td className="num muted">{x.first ?? "—"} → {x.last ?? "—"}</td>
                <td className="num">{x.ratio != null ? pct(x.ratio) : "—"}</td>
                <td className="num">{x.same != null ? pct(x.same) : "—"}</td>
                <td className="num">{x.models}</td>
                <td className="mono clip" title={x.top?.[0]}>{x.top ? <>{x.top[0]} <span className="muted">{x.top[1]}/{x.r.n_ok}</span></> : "—"}</td>
                <td className="num">{x.gpu != null ? pct(x.gpu) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

// --------------------------------------------------------------------------- //
// page
// --------------------------------------------------------------------------- //

export function DashboardView({ data }: { data: Dashboard }) {
  const datasets = useMemo(() => [...new Set(data.runs.map((r) => r.dataset_label))], [data]);
  const agents = useMemo(() => [...new Set(data.runs.map((r) => agentName(r.agent)))].sort(), [data]);
  const [dsOn, setDsOn] = useState(() => new Set(datasets));
  const [agOn, setAgOn] = useState(() => new Set(agents));
  const color = useAgentColor(agents);
  const runs = data.runs.filter((r) => dsOn.has(r.dataset_label) && agOn.has(agentName(r.agent)));
  const shownAgents = agents.filter((a) => runs.some((r) => agentName(r.agent) === a));

  const steps = runs.flatMap((r) => r.steps);
  const ratios = steps.map((s) => s.ratio).filter((x): x is number => x != null);
  const diffs = steps.filter((s) => s.same != null);

  return (
    <>
      <div className="selection dash-filters">
        <span className="small muted">datasets</span>
        <Chips all={datasets} on={dsOn} set={setDsOn} count={(d) => data.runs.filter((r) => r.dataset_label === d).length} />
        <span className="small muted">agents</span>
        <Chips all={agents} on={agOn} set={setAgOn} color={(a) => color(a)}
               count={(a) => data.runs.filter((r) => agentName(r.agent) === a).length} />
      </div>

      <div className="facts dash-kpis">
        <div><b>{runs.length}</b><span className="muted">runs</span></div>
        <div><b>{new Set(runs.map((r) => r.dataset)).size}</b><span className="muted">datasets</span></div>
        <div><b>{steps.length.toLocaleString()}</b><span className="muted">scripts</span></div>
        <div><b>{median(steps.filter((s) => s.ok).map((s) => s.loc))?.toFixed(0) ?? "—"}</b><span className="muted">median lines of code</span></div>
        <div><b>{ratios.length ? pct(median(ratios)!) : "—"}</b><span className="muted">median change per step</span></div>
        <div><b>{diffs.length ? pct(diffs.filter((s) => s.same).length / diffs.length) : "—"}</b><span className="muted">steps repeating the parent</span></div>
      </div>

      {runs.length === 0 ? <p className="muted">No run matches the filters.</p> : (
        <div className="dash-grid">
          <CdfPanel runs={runs} color={color} agents={shownAgents} />
          <AlongPanel runs={runs} color={color} agents={shownAgents} />
          <Heatmap runs={runs} color={color} />
          <ExplorePanel runs={runs} color={color} />
          <SizePanel runs={runs} color={color} />
          <RunTable runs={runs} color={color} />
        </div>
      )}
    </>
  );
}

