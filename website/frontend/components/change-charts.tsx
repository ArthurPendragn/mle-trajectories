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

const H = 110;                       // plot height
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
    <figure className="chg-chart">
      <figcaption>{title}</figcaption>
      <svg width={width} height={H + M.top + M.bottom} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={M.left + plotW} y1={y(t)} y2={y(t)} className="chg-grid" />
            <text x={M.left - 6} y={y(t) + 3} className="chg-tick" textAnchor="end">
              {Math.round(t).toLocaleString()}
            </text>
          </g>
        ))}
        {hover !== null && (
          <rect x={M.left + hover * band} y={M.top} width={band} height={H} className="chg-hover" />
        )}
        {points.map((p, i) => {
          const v = value(p);
          const x = M.left + i * band + (band - barW) / 2;
          if (empty?.(p)) {
            return <circle key={p.name} cx={x + barW / 2} cy={base - 4} r={Math.min(4, Math.max(2, barW / 2))} className="chg-empty" />;
          }
          return v == null ? null : <path key={p.name} d={column(x, barW, y(v), base)} className="chg-bar" />;
        })}
        {(bestI < 0 || (bestI + 0.5) * band > 50) && <text x={M.left} y={base + 13} className="chg-tick">step 1</text>}
        {(bestI < 0 || plotW - (bestI + 0.5) * band > 50) && (
          <text x={M.left + plotW} y={base + 13} className="chg-tick" textAnchor="end">step {points.length}</text>
        )}
        {bestI >= 0 && (
          <text x={M.left + (bestI + 0.5) * band} y={base + 13} className="chg-tick chg-best" textAnchor="middle">▲ best</text>
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
    <div className="chg" ref={box}>
      <Columns title="Lines changed against the parent (added + removed; ○ same code, no bar: no parent)"
               points={points} value={(p) => p.diff?.lines_changed ?? null} width={width}
               hover={hover} setHover={setHover} best={best}
               empty={(p) => !!p.diff?.same_code} />
      <Columns title="Lines of code" points={points} value={(p) => p.loc} width={width}
               hover={hover} setHover={setHover} best={best} />
      {h && (
        <div className="chg-tip" style={{ left: Math.min(Math.max(left, 90), width - 90) }}>
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
