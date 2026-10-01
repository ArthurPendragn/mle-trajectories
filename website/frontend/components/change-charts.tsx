"use client";

import { useEffect, useRef, useState } from "react";
import { fmtScore } from "@/lib/format";
import type { CodeDiff } from "@/lib/types";

type Point = {
  name: string;
  i: number;
  loc: number | null;
  diff: CodeDiff | null;
  score: number | null;
};
const TIP: React.CSSProperties = {
  position: "absolute", top: 28, transform: "translateX(-50%)", pointerEvents: "none", zIndex: 3,
  background: "var(--surface)", color: "var(--fg)", border: "1px solid var(--border)", borderRadius: 6,
  padding: "6px 9px", fontSize: 12, boxShadow: "0 2px 8px rgba(0,0,0,.12)", whiteSpace: "nowrap",
};

const H = 110;                       // plot height
// marks are coloured by CSS (theme-aware); the attribute is only the fallback,
// so a stale stylesheet shows blue marks rather than SVG's default black
const FALLBACK = "#6f8fd0";
// every colour inline, from the theme variables (defined since the first
// stylesheet), so marks and labels follow light/dark mode even when the
// browser holds an older copy of the chart's CSS rules
const S = {
  grid: { stroke: "var(--border)", strokeWidth: 1 },
  axis: { stroke: "var(--muted)", strokeWidth: 1 },
  tick: { fill: "var(--muted)", fontSize: 10 },
  strong: { fill: "var(--fg)", fontSize: 10 },
  bar: { fill: "var(--accent)" },
  empty: { fill: "var(--surface)", stroke: "var(--accent)", strokeWidth: 1.5 },
  hover: { fill: "var(--badge)" },
  line: { stroke: "var(--accent)", strokeWidth: 2, strokeLinejoin: "round" as const, fill: "none" },
  ref: { stroke: "var(--muted)", strokeWidth: 1.2 },
  dot: { fill: "var(--accent)", stroke: "var(--surface)", strokeWidth: 2 },
};
const M = { top: 8, right: 8, bottom: 18, left: 44 };

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v) ?? v;
}

/** A column with a 4px rounded top, square at the baseline. */
function column(x: number, w: number, y: number, base: number): string {
  const h = base - y;
  if (h <= 0) return "";
  const r = Math.min(4, w / 2, h);
  return `M${x},${base}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${base}Z`;
}

function Columns({ title, points, value, width, hover, setHover, best, empty }: {
  title: string;
  points: Point[];
  value: (p: Point) => number | null;
  width: number;
  hover: number | null;
  setHover: (i: number | null) => void;
  best: string | null;
  empty?: (p: Point) => boolean;            // drawn as a hollow marker on the baseline
}) {
  const plotW = Math.max(width - M.left - M.right, 10);
  const band = plotW / Math.max(points.length, 1);
  const barW = Math.max(1, Math.min(24, band - 2));
  const max = niceMax(Math.max(0, ...points.map((p) => value(p) ?? 0)));
  const y = (v: number) => M.top + H - (v / max) * H;
  const base = M.top + H;
  const ticks = [0, max / 2, max];
  const bestI = points.findIndex((p) => p.name === best);
  return (
    <figure className="chg-chart" style={{ margin: "0 0 6px" }}>
      <figcaption style={{ fontSize: 12, color: "var(--muted)", margin: "0 0 2px" }}>{title}</figcaption>
      <svg width={width} height={H + M.top + M.bottom} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} className="chg-grid" style={S.grid} />
            <text x={M.left - 6} y={y(t) + 3} className="chg-tick" style={S.tick} textAnchor="end">
              {Math.round(t).toLocaleString()}
            </text>
          </g>
        ))}
        {hover !== null && (
          <rect x={M.left + hover * band} y={M.top} width={band} height={H} className="chg-hover" style={S.hover} />
        )}
        {points.map((p, i) => {
          const v = value(p);
          const x = M.left + i * band + (band - barW) / 2;
          if (empty?.(p)) {
            return <circle key={p.name} cx={x + barW / 2} cy={base - 4} r={Math.min(4, Math.max(2, barW / 2))}
                           className="chg-empty" style={S.empty} />;
          }
          return v == null ? null : <path key={p.name} d={column(x, barW, y(v), base)} className="chg-bar" fill={FALLBACK} style={S.bar} />;
        })}
        {(bestI < 0 || (bestI + 0.5) * band > 50) && <text x={M.left} y={base + 13} className="chg-tick" style={S.tick}>step 1</text>}
        {(bestI < 0 || plotW - (bestI + 0.5) * band > 50) && (
          <text x={M.left + plotW} y={base + 13} className="chg-tick" style={S.tick} textAnchor="end">step {points.length}</text>
        )}
        {bestI >= 0 && (
          <text x={M.left + (bestI + 0.5) * band} y={base + 13} className="chg-tick chg-best" style={S.strong} textAnchor="middle">▲ best</text>
        )}
        {points.map((p, i) => (
          <rect key={p.name} x={M.left + i * band} y={M.top} width={band} height={H + M.bottom}
                fill="transparent" onMouseEnter={() => setHover(i)} />
        ))}
      </svg>
    </figure>
  );
}

/** How the code changes along the trajectory: lines changed against the
 *  parent, and the size of each step, as two aligned column charts. */
export function ChangeCharts({ points, best }: { points: Point[]; best: string | null }) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    if (!box.current) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.floor(e.contentRect.width)));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const h = hover === null ? null : points[hover];
  const left = hover === null ? 0 : M.left + (hover + 0.5) * ((width - M.left - M.right) / points.length);

  return (
    <div className="chg" ref={box} style={{ position: "relative", margin: "4px 0 8px" }}>
      <Columns title="Lines changed against the parent (added + removed; ○ same code, no bar: no parent)"
               points={points} value={(p) => p.diff?.lines_changed ?? null} width={width}
               hover={hover} setHover={setHover} best={best}
               empty={(p) => !!p.diff?.same_code} />
      <Columns title="Lines of code" points={points} value={(p) => p.loc} width={width}
               hover={hover} setHover={setHover} best={best} />
      {h && (
        <div className="chg-tip" style={{ ...TIP, left: Math.min(Math.max(left, 90), width - 90) }}>
          <b className="mono">{h.name}</b> <span className="muted">step {h.i + 1}</span>
          <div>
            {h.diff
              ? h.diff.same_code ? "same code as its parent"
                : <>+{h.diff.lines_added} / −{h.diff.lines_removed} lines · {Math.round(h.diff.similarity * 100)}% similar</>
              : <span className="muted">no parent</span>}
          </div>
          <div>{h.loc ?? "—"} lines of code{h.score != null && <> · score {fmtScore(h.score)}</>}</div>
        </div>
      )}
    </div>
  );
}

export type { Point as ChangePoint };

/** numpy.percentile(..., method="linear") on sorted values. */
function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Empirical CDF of the ratio of changed code lines over every parent -> child
 *  step, with the median, p90 and p99 dropped to both axes. */
export function ChangeCdf({ ratios }: { ratios: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 390, PH = 170, m = { top: 10, right: 22, bottom: 30, left: 40 };
  const pw = W - m.left - m.right;
  const n = ratios.length;
  if (n === 0) return null;
  const xmax = Math.max(1, Math.ceil(ratios[n - 1] * 4) / 4);
  const x = (r: number) => m.left + (r / xmax) * pw;
  const y = (f: number) => m.top + PH - f * PH;
  let d = `M${x(0)},${y(0)}`;
  ratios.forEach((r, i) => { d += `H${x(r)}V${y((i + 1) / n)}`; });
  d += `H${x(xmax)}`;
  const marks = [
    { q: 0.5, label: "median", dash: undefined },
    { q: 0.9, label: "p90", dash: "5 3" },
    { q: 0.99, label: "p99", dash: "1.5 2.5" },
  ].map((k) => ({ ...k, v: quantile(ratios, k.q) }));
  const xticks = Array.from({ length: Math.round(xmax / 0.25) + 1 }, (_, i) => i * 0.25)
    .filter((t, i, a) => a.length <= 6 || i % 2 === 0);
  const below = (r: number) => ratios.filter((v) => v <= r).length;

  function move(e: React.MouseEvent<SVGRectElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const r = ((e.clientX - box.left) / box.width) * xmax;
    setHover(Math.max(0, Math.min(xmax, r)));
  }

  return (
    <div className="cdf" style={{ display: "flex", flexWrap: "wrap", gap: "8px 24px", alignItems: "flex-start", margin: "4px 0 12px" }}>
      <svg width={W} height={PH + m.top + m.bottom}>
        {[0.5, 0.9, 0.99].map((f) => (
          <line key={f} x1={m.left} x2={m.left + pw} y1={y(f)} y2={y(f)} className="chg-grid" style={S.grid} />
        ))}
        <line x1={m.left} x2={m.left + pw} y1={y(0)} y2={y(0)} className="chg-axis" style={S.axis} />
        {marks.map((k) => (
          <g key={k.label} className="cdf-ref" style={S.ref}>
            <line x1={x(k.v)} x2={x(k.v)} y1={y(0)} y2={y(k.q)} strokeDasharray={k.dash} />
            <line x1={m.left} x2={x(k.v)} y1={y(k.q)} y2={y(k.q)} strokeDasharray={k.dash} />
          </g>
        ))}
        <path d={d} className="cdf-line" style={S.line} />
        {[0.5, 0.9, 0.99].map((f) => (
          <text key={f} x={m.left - 6} y={y(f) + 3} className="chg-tick" style={S.tick} textAnchor="end">{pct(f)}</text>
        ))}
        {xticks.map((t) => (
          <text key={t} x={x(t)} y={y(0) + 13} className="chg-tick" style={S.tick} textAnchor="middle">{pct(t)}</text>
        ))}
        <text x={m.left + pw / 2} y={y(0) + 27} className="chg-tick" style={S.tick} textAnchor="middle">
          ratio of changed code lines (against the parent)
        </text>
        <text x={12} y={m.top + PH / 2} className="chg-tick" style={S.tick} textAnchor="middle"
              transform={`rotate(-90 12 ${m.top + PH / 2})`}>CDF</text>
        {hover !== null && (
          <g>
            <line x1={x(hover)} x2={x(hover)} y1={m.top} y2={y(0)} className="chg-cross" style={S.axis} />
            <circle cx={x(hover)} cy={y(below(hover) / n)} r={4} className="cdf-dot" style={S.dot} />
          </g>
        )}
        <rect x={m.left} y={m.top} width={pw} height={PH} fill="transparent"
              onMouseMove={move} onMouseLeave={() => setHover(null)} />
      </svg>
      <div className="cdf-side small">
        <p className="muted">{n} parent → child steps</p>
        {marks.map((k) => (
          <p key={k.label}>
            <svg width="22" height="8" className="cdf-key"><line x1="0" x2="22" y1="4" y2="4" style={S.ref}
              strokeDasharray={k.dash} strokeWidth="1.5" /></svg>
            {k.label} <b>{pct(k.v)}</b>
          </p>
        ))}
        {hover !== null && (
          <p className="cdf-read">
            <b>{pct(below(hover) / n)}</b> of steps changed at most <b>{pct(hover)}</b> of their parent&apos;s code lines
          </p>
        )}
      </div>
    </div>
  );
}
