import type { Metric } from "./types";

export function fmtScore(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x.toPrecision(5)).toString();
}

/** Signed change of `score` over `parent`, oriented so positive = better. */
export function improvement(score: number | null, parent: number | null,
                            metric: Metric): number | null {
  if (score === null || parent === null) return null;
  const d = score - parent;
  return metric.lower_is_better ? -d : d;
}

export function metricLabel(m: Metric): string {
  const arrow = m.lower_is_better === true ? " ↓" : m.lower_is_better === false ? " ↑" : "";
  return `${m.name ?? "metric ?"}${arrow}`;
}

export function shortCommit(c: string | null | undefined): string {
  return !c ? "?" : c === "unknown" ? "unknown" : c.slice(0, 7);
}

export function firstLine(text: string | null, limit = 140): string {
  if (!text) return "";
  const line = text.trim().split("\n")[0];
  return line.length > limit ? `${line.slice(0, limit - 1)}…` : line;
}

export function fmtSecs(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  if (s < 1) return `${(s * 1000).toFixed(s < 0.01 ? 2 : 0)} ms`;
  if (s < 90) return `${s.toFixed(1)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export function fmtNum(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined) return "—";
  return Number.isInteger(x) ? String(x) : x.toFixed(digits);
}
