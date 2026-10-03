import { fmtScore, fmtSigned, gap } from "@/lib/format";
import type { Metric, TestScores } from "@/lib/types";

const HOW = "not scored yet: add test_scores.toml to the run folder (see website/README.md)";

/** The final submission's score on the held-out test labels, with its gap to the
 *  same step's validation score (positive = test better than validation). */
export function TestScore({ test, metric, detail = false }:
  { test: TestScores | null; metric: Metric; detail?: boolean }) {
  const f = test?.final;
  if (!test || !f) return <span className="muted" title={HOW}>—</span>;
  const g = test.metric ? null : gap(f.score, f.val_score, metric);   // other metric: no gap
  const title = [
    f.pipeline && `submitted: ${f.pipeline}`,
    f.val_score != null && `validation ${fmtScore(f.val_score)}`,
    test.metric && `test metric: ${test.metric}`,
    f.file && `predictions: ${f.file}`,
    test.scored_at && `scored ${test.scored_at}`,
    test.note, f.note,
  ].filter(Boolean).join("\n");
  return (
    <span title={title}>
      {fmtScore(f.score)}
      {test.metric && !detail && <span className="muted small"> {test.metric}</span>}
      {g !== null && <span className={`small ${g >= 0 ? "up" : "down"}`}> {fmtSigned(g)}</span>}
      {detail && f.pipeline && <> <span className="mono small muted">{f.pipeline}</span></>}
    </span>
  );
}
