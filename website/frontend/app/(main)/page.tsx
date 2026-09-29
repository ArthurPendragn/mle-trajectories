import Link from "next/link";
import { getCorpus } from "@/lib/dal";
import { fmtScore, metricLabel, shortCommit } from "@/lib/format";
import type { DatasetInfo, RunSummary } from "@/lib/types";
import { StaleHint } from "@/components/stale-hint";

export const dynamic = "force-dynamic";

function DataBadge({ ds }: { ds: DatasetInfo }) {
  const title = ds.data_status === "remote" ? ds.data
    : ds.data_status === "missing" ? "input/ not on this machine" : "input/ present";
  return (
    <span className={`badge data-${ds.data_status}`} title={title}>
      data: {ds.data_status}
      {ds.samples.length > 0 && ` · ${ds.samples.length} sample${ds.samples.length > 1 ? "s" : ""}`}
    </span>
  );
}

function Plans({ run }: { run: RunSummary }) {
  const src = run.default_source;
  if (!src) return <span className="muted">—</span>;
  const [have, total] = src.coverage;
  const extra = run.n_sources - 1;
  return (
    <span title={src.dirs.join(", ")}>
      <span className={have === total ? "" : "partial"}>{have}/{total}</span>{" "}
      <span className="muted">{src.skrub && src.name === "pipelines" ? "agent" : src.name}</span>
      {extra > 0 && <span className="muted"> +{extra}</span>}
    </span>
  );
}

function Runtime({ run, current }: { run: RunSummary; current: string | null }) {
  if (run.runtime.length === 0) return <span className="muted">—</span>;
  return (
    <span className="stack">
      {run.runtime.map((r) => (
        <span key={r.file}>
          {r.n_ok} ok{r.n_failed > 0 && <span className="muted"> · {r.n_failed} failed</span>}
          {r.sample_rows && <span className="muted"> @{r.sample_rows.toLocaleString()} rows</span>}{" "}
          <StaleHint store={r} current={current} compact />
        </span>
      ))}
    </span>
  );
}

export default async function CorpusPage() {
  const corpus = await getCorpus();
  const runs = corpus.datasets.flatMap((d) => d.runs);
  const steps = runs.reduce((n, r) => n + r.n_steps, 0);
  return (
    <>
      <h1>Corpus</h1>
      <p className="muted">
        {corpus.datasets.length} datasets · {runs.length} runs · {steps} steps ·
        stratum {shortCommit(corpus.stratum_commit)}
      </p>
      {corpus.datasets.map((ds) => (
        <section key={ds.name} className="dataset">
          <div className="dataset-head">
            <h2>{ds.label}</h2>
            <span className="muted">{ds.task}</span>
            <span className="badge">{metricLabel(ds.metric)}</span>
            <DataBadge ds={ds} />
          </div>
          <table>
            <thead>
              <tr>
                <th>Run</th><th>Agent</th><th className="num">Steps</th>
                <th className="num">Best</th><th>Lineage</th><th>Skrub plans</th><th>Runtime</th>
              </tr>
            </thead>
            <tbody>
              {ds.runs.map((r) => (
                <tr key={r.id}>
                  <td>
                    <Link href={`/runs/${r.dataset}/${r.name}`}>{r.label}</Link>
                    <div className="muted small">{r.name}</div>
                  </td>
                  <td><span className={`badge agent-${r.agent}`}>{r.agent ?? "?"}</span></td>
                  <td className="num">{r.n_steps || <span className="muted">—</span>}</td>
                  <td className="num" title={r.best?.module ?? undefined}>{fmtScore(r.best?.score)}</td>
                  <td>{r.lineage ?? <span className="muted">none</span>}</td>
                  <td><Plans run={r} /></td>
                  <td><Runtime run={r} current={corpus.stratum_commit} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}
    </>
  );
}
