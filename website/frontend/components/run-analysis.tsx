"use client";

import { useEffect, useRef, useState } from "react";
import { Explorer, type ExplorerHandle } from "./explorer/explorer";
import { fmtNum, fmtSecs, shortCommit } from "@/lib/format";
import type { AnalysisData, AnalysisStatus, OpStats, TreeData } from "@/lib/types";

const POLL_MS = 2000;

function useAnalysis(dataset: string, run: string, source: string | null,
                     initial: AnalysisStatus | null) {
  const [state, setState] = useState<AnalysisStatus | null>(initial);
  const [retrying, setRetrying] = useState(false);
  useEffect(() => setState(initial), [initial]);

  const url = source
    ? `/api/analysis/${encodeURIComponent(dataset)}/${encodeURIComponent(run)}/${encodeURIComponent(source)}`
    : null;

  useEffect(() => {
    if (!url || !state || state.status === "ready" || state.status === "failed") return;
    const t = setTimeout(async () => {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (res.ok) setState(await res.json());
      } catch {
        /* next tick retries */
      }
    }, POLL_MS);
    return () => clearTimeout(t);
  }, [url, state]);

  async function retry() {
    if (!url) return;
    setRetrying(true);
    try {
      const res = await fetch(`${url}?retry=1`, { cache: "no-store" });
      if (res.ok) setState(await res.json());
    } finally {
      setRetrying(false);
    }
  }
  return { state, retry, retrying };
}

// --------------------------------------------------------------------------- //
function SearchTree({ tree, selected, onNode }: {
  tree: TreeData;
  selected: Set<string>;
  onNode: (module: string, withSubtree: boolean) => void;
}) {
  const modes = (["delta", "phase"] as const).filter((m) => tree.svg[m]);
  const [mode, setMode] = useState<"delta" | "phase">("delta");
  const [fit, setFit] = useState(true);   // scale to the box width, or graphviz's own size
  const box = useRef<HTMLDivElement>(null);

  // selection outline, driven by the explorer's ticked pipelines
  useEffect(() => {
    box.current?.querySelectorAll<SVGGElement>("g[data-module]").forEach((g) => {
      g.classList.toggle("sel", selected.has(g.dataset.module ?? ""));
    });
  }, [selected, mode]);

  function click(e: React.MouseEvent) {
    const g = (e.target as Element).closest("g[data-module]");
    const module = g?.getAttribute("data-module");
    if (module) onNode(module, e.shiftKey);
  }

  return (
    <>
      <p className="muted small">
        Every step the agent ran, linked to the step it was derived from. Fill: green improved on
        the parent · amber flat · red regressed{tree.lower_is_better ? " (lower is better)" : ""}.
        Dashed: no plan in the selected source. Click a node to tick it in the explorer;
        shift-click takes its whole subtree.
      </p>
      <div className="tree-modes">
        <button className="link-button" onClick={() => setFit(!fit)}>
          {fit ? "actual size" : "fit width"}
        </button>
        {modes.length > 1 && <>
          colour by:
          {modes.map((m) => (
            <label key={m}>
              <input type="radio" checked={mode === m} onChange={() => setMode(m)} />
              {m === "delta" ? "score vs parent" : "search phase"}
            </label>
          ))}
          {mode === "phase" && (
            <span className="legend">
              {Object.entries(tree.phase_colors).map(([ph, c]) => (
                <span key={ph}><span className="dot" style={{ background: c }} />{ph}</span>
              ))}
            </span>
          )}
        </>}
      </div>
      <div className={`tree-canvas${fit ? " fit" : ""}`} ref={box} onClick={click}
           dangerouslySetInnerHTML={{ __html: tree.svg[mode] ?? "" }} />
    </>
  );
}

// --------------------------------------------------------------------------- //
function StatsTable({ stats }: { stats: OpStats }) {
  return (
    <table>
      <thead>
        <tr>
          <th>operator</th><th className="num">total</th><th className="num">present</th>
          <th className="num">mean</th><th className="num">median</th><th className="num">std</th>
          <th className="num">min</th><th className="num">max</th>
        </tr>
      </thead>
      <tbody>
        {stats.rows.map((r) => (
          <tr key={r.op}>
            <td className="mono small">{r.op}</td>
            <td className="num">{r.total}</td>
            <td className="num">{r.present}/{stats.n_pipelines}</td>
            <td className="num">{fmtNum(r.mean)}</td>
            <td className="num">{fmtNum(r.median, 1)}</td>
            <td className="num">{fmtNum(r.std)}</td>
            <td className="num">{r.min}</td>
            <td className="num">{r.max}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StatsSummary({ stats }: { stats: OpStats }) {
  if (!stats.sizes) return null;
  return (
    <p className="muted small">
      {stats.n_pipelines} pipelines · {stats.rows.length} operator types ·{" "}
      {stats.sizes.total} operator instances · DAG size per pipeline: median{" "}
      {stats.sizes.median}, min {stats.sizes.min}, max {stats.sizes.max}. Distribution
      columns span all pipelines (a pipeline lacking an operator counts as 0).
    </p>
  );
}

function OperatorStats({ data }: { data: AnalysisData }) {
  const { logical, physical, physical_missing } = data.stats;
  return (
    <>
      <h3>Logical IR</h3>
      <StatsSummary stats={logical} />
      <StatsTable stats={logical} />
      {physical.n_pipelines > 0 && (
        <details className="block">
          <summary>
            Physical, default selection ({physical.rows.length} operator kinds
            {physical_missing > 0 && `, ${physical_missing} pipeline(s) without a physical plan`})
          </summary>
          <p className="muted small">
            Physical lowering plus default implementation selection; operators are split by
            operation kind (e.g. NumericOp[square]).
          </p>
          <StatsTable stats={physical} />
        </details>
      )}
    </>
  );
}

// --------------------------------------------------------------------------- //
function Pending({ state, retry, retrying }: {
  state: AnalysisStatus;
  retry: () => void;
  retrying: boolean;
}) {
  if (state.status === "failed") {
    return (
      <div className="pending failed">
        <p>Building the operator graphs failed.</p>
        <pre className="log">{state.log}</pre>
        <button onClick={retry} disabled={retrying}>{retrying ? "starting…" : "retry"}</button>
      </div>
    );
  }
  const what = state.status === "queued" ? "Queued — other analyses are building"
    : "Building the operator graphs (imports every pipeline, no data read)…";
  return (
    <div className="pending">
      <p><span className="spinner" /> {what}</p>
      {"log" in state && state.log && <pre className="log">{state.log}</pre>}
    </div>
  );
}

export function RunAnalysis({ dataset, run, source, noSourceReason, tree, initial }: {
  dataset: string;
  run: string;
  source: string | null;
  noSourceReason: string;
  tree: TreeData;
  initial: AnalysisStatus | null;
}) {
  const { state, retry, retrying } = useAnalysis(dataset, run, source, initial);
  const explorer = useRef<ExplorerHandle>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const data = state?.status === "ready" ? state.data : null;

  function onNode(module: string, withSubtree: boolean) {
    if (!explorer.current?.toggleName(module, withSubtree)) {
      document.getElementById(`step-${module}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  return (
    <>
      <section>
        <h2>Search tree</h2>
        {tree.modules.length === 0 ? <p className="muted">No lineage recorded for this run.</p>
          : <SearchTree tree={tree} selected={selected} onNode={onNode} />}
      </section>

      <section>
        <h2>Operator explorer</h2>
        {!source ? <p className="muted">{noSourceReason}</p>
          : !state ? <p className="muted">Unavailable.</p>
          : !data ? <Pending state={state} retry={retry} retrying={retrying} />
          : (
            <>
              <p className="muted small">
                One graph for the whole run: {data.merged.pipelines.filter((p) => p.ok).length}{" "}
                pipelines merge into <b>{data.merged.nodes.length}</b> distinct operations — an
                operation shared by several pipelines is one node, keyed by the content signature of
                its whole sub-computation. Built from <span className="mono">{data.source}</span> in{" "}
                {fmtSecs(data.elapsed_s)} under stratum {shortCommit(data.stratum_commit)}
                {data.folded.length > 0 && `; ${data.folded.length} code-identical step(s) share one DAG`}.
              </p>
              {data.failed.length > 0 && (
                <details className="block">
                  <summary className="partial">
                    {data.failed.length} of {data.n_pipelines} pipeline(s) could not be extracted
                  </summary>
                  {data.failed.map((f) => (
                    <div key={f.name}>
                      <b className="mono small">{f.name}</b>
                      <pre className="log">{f.error}</pre>
                    </div>
                  ))}
                </details>
              )}
              <Explorer ref={explorer} data={data.merged}
                        onSelect={(names) => setSelected(new Set(names))} />
            </>
          )}
      </section>

      {data && (
        <section>
          <h2>Operator statistics</h2>
          <OperatorStats data={data} />
        </section>
      )}
    </>
  );
}
