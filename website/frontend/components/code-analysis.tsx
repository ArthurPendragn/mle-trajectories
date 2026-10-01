"use client";

import { Fragment, useState } from "react";
import type { CodeAnalysis, CodeFeatures, Component } from "@/lib/types";

// Kinds shown by default; the rest are building blocks (layers, data loaders, ...)
const MAIN_KINDS = new Set(["model", "ensemble", "transformer", "pipeline", "splitter", "search", "metric"]);

function Dist({ d, label }: { d: { median: number; min: number; max: number } | null; label: string }) {
  if (!d) return null;
  return <span>{label} <b>{d.median}</b> <span className="muted">({d.min}–{d.max})</span></span>;
}

function Bar({ n, of }: { n: number; of: number }) {
  return (
    <span className="share" title={`${n} of ${of}`}>
      <span className="share-bar"><span style={{ width: `${(100 * n) / Math.max(of, 1)}%` }} /></span>
      {n}/{of}
    </span>
  );
}

/** A parameter value: a literal as written, an expression (``=...``) marked. */
function Param({ v }: { v: unknown }) {
  if (typeof v === "string" && v.startsWith("=")) return <span className="expr">{v.slice(1)}</span>;
  return <>{typeof v === "string" ? JSON.stringify(v) : JSON.stringify(v)}</>;
}

function Params({ c }: { c: Component }) {
  const entries = Object.entries(c.params);
  if (entries.length === 0) return <span className="muted">—</span>;
  return (
    <>
      {entries.map(([k, v], i) => (
        <span key={k}>{i > 0 && ", "}<span className="muted">{k}=</span><Param v={v} /></span>
      ))}
    </>
  );
}

function names(p: CodeFeatures, kind: string): string[] {
  return [...new Set((p.components ?? []).filter((c) => c.kind === kind).map((c) => c.name))];
}

function Detail({ p }: { p: CodeFeatures }) {
  if (!p.ok) return <p className="error small">Does not parse: {p.error}</p>;
  const comps = (p.components ?? []).filter((c) => c.kind !== "layer");
  const d = p.data!;
  return (
    <div className="code-detail">
      <p className="mono muted small">{p.file}</p>
      {comps.length > 0 && (
        <table className="small">
          <thead><tr><th>line</th><th>kind</th><th>component</th><th>parameters</th></tr></thead>
          <tbody>
            {comps.map((c, i) => (
              <tr key={i}>
                <td className="num muted">{c.line ?? ""}</td>
                <td className="muted">{c.kind}</td>
                <td className="mono" title={c.qualified}>{c.name}</td>
                <td className="mono">{c.network ? <span className="muted">defined in the file</span> : <Params c={c} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <dl className="meta small">
        <div><dt>reads</dt><dd className="mono">{d.reads.map((r) => r.path ?? `${r.func}(…)`).join(", ") || "—"}</dd></div>
        <div><dt>writes</dt><dd className="mono">{d.writes.map((r) => r.path ?? `${r.func}(…)`).join(", ") || "—"}</dd></div>
        <div><dt>column writes</dt><dd>{d.column_writes} <span className="muted">({d.columns_written} distinct names) · .loc/.iloc {d.loc_writes} · inplace {d.inplace}</span></dd></div>
        <div><dt>pandas</dt><dd className="mono">{Object.entries(d.pandas).map(([k, v]) => `${k} ${v}`).join(", ") || "—"}</dd></div>
        <div><dt>polars</dt><dd className="mono">{Object.entries(d.polars).map(([k, v]) => `${k} ${v}`).join(", ") || "—"}</dd></div>
        <div><dt>imports</dt><dd className="mono">{p.imports!.join(", ")}</dd></div>
        <div><dt>other</dt><dd>
          {p.other!.gpu ? "GPU · " : ""}{p.other!.pip_installs.length > 0 && `pip install ${p.other!.pip_installs.join(" ")} · `}
          {p.other!.shell_calls > 0 && `${p.other!.shell_calls} shell call(s) · `}
          seeds {p.other!.seeds.join(", ") || "—"} · {p.other!.prints} print(s)
        </dd></div>
      </dl>
    </div>
  );
}

export function CodeAnalysisView({ data, best }: { data: CodeAnalysis; best: string | null }) {
  const [blocks, setBlocks] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  if (data.covered_by || !data.summary || !data.pipelines) {
    return <p className="muted">{data.reason}</p>;
  }
  const s = data.summary;
  const n = s.n - s.n_failed;
  const comps = s.components.filter((c) => blocks || MAIN_KINDS.has(c.kind));

  return (
    <>
      <p className="muted small">
        Read from the agent&apos;s original scripts, without running or importing them
        (<span className="mono">tools/code_stats</span>): what was <i>written</i>, so a branch that never
        ran still counts. Methods are counted by name, limited to names specific to pandas or polars.
      </p>
      <p className="code-facts">
        <span><b>{s.n}</b> scripts{s.n_failed > 0 && <span className="partial"> ({s.n_failed} do not parse)</span>}</span>
        <Dist d={s.loc} label="lines of code" />
        <Dist d={s.complexity} label="complexity" />
        <Dist d={s.loops} label="loops" />
        <Dist d={s.ifs} label="if" />
        <Dist d={s.tries} label="try" />
        <Dist d={s.functions} label="functions" />
        <Dist d={s.classes} label="classes" />
        <Dist d={s.column_writes} label="df[…] = writes" />
      </p>

      <div className="code-grid">
        <div>
          <h3>
            Components{" "}
            <label className="small muted"><input type="checkbox" checked={blocks}
              onChange={(e) => setBlocks(e.target.checked)} /> building blocks too (layers, losses, optimizers, data loaders)</label>
          </h3>
          <table className="small">
            <colgroup><col style={{ width: "6.5em" }} /><col /><col style={{ width: "5.5em" }} />
              <col style={{ width: "9.5em" }} /><col style={{ width: "11em" }} /></colgroup>
            <thead><tr><th>kind</th><th>component</th><th>library</th><th>pipelines</th><th>first in</th></tr></thead>
            <tbody>
              {comps.map((c) => (
                <tr key={`${c.kind}:${c.name}`}>
                  <td className="muted">{c.kind}</td>
                  <td className="mono clip" title={c.name}>{c.name}</td>
                  <td className="muted">{c.lib}</td>
                  <td><Bar n={c.n} of={n} /></td>
                  <td className="mono muted clip" title={c.first}>{c.first}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3>Libraries</h3>
          <table className="small"><tbody>
            {s.libraries.map((l) => (
              <tr key={l.name}><td className="mono">{l.name}</td><td><Bar n={l.n} of={n} /></td></tr>
            ))}
          </tbody></table>
          {s.splitters.length > 0 && <>
            <h3>Validation</h3>
            <table className="small"><tbody>
              {s.splitters.map((x) => <tr key={x.name}><td className="mono">{x.name}</td><td><Bar n={x.n} of={n} /></td></tr>)}
            </tbody></table>
          </>}
          <h3>Files read</h3>
          <table className="small"><tbody>
            {s.reads.slice(0, 12).map((r) => <tr key={r.path}><td className="mono clip" title={r.path}>{r.path}</td><td><Bar n={r.n} of={n} /></td></tr>)}
          </tbody></table>
          <p className="small muted">
            GPU in {s.gpu}/{n} · <span className="mono">inplace=True</span> in {s.inplace}/{n} · shell calls in {s.shell}/{n}
            {s.pip_installs.length > 0 && <> · <span className="partial">pip install at run time</span>:{" "}
              {s.pip_installs.slice(0, 6).map((p) => `${p.name} (${p.n})`).join(", ")}</>}
          </p>
          {(Object.keys(s.pandas).length > 0 || Object.keys(s.polars).length > 0) && (
            <p className="small wrap">
              <span className="muted">pandas:</span>{" "}
              <span className="mono">{Object.entries(s.pandas).slice(0, 10).map(([k, v]) => `${k} ${v}`).join(", ") || "—"}</span>
              <br />
              <span className="muted">polars:</span>{" "}
              <span className="mono">{Object.entries(s.polars).slice(0, 10).map(([k, v]) => `${k} ${v}`).join(", ") || "—"}</span>
            </p>
          )}
        </div>
      </div>

      <h3>Per pipeline <span className="muted small">click a row for its components, parameters and data handling</span></h3>
      <div className="table-scroll">
        <table className="small code-table">
          <thead>
            <tr>
              <th>pipeline</th><th className="num">loc</th><th className="num">cx</th>
              <th className="num">for</th><th className="num">if</th><th className="num">fn</th>
              <th>models</th><th>transformers</th><th>validation</th><th>metric</th>
              <th className="num">df[…]=</th><th className="num">reads</th>
            </tr>
          </thead>
          <tbody>
            {data.pipelines.map((p) => (
              <Fragment key={p.name}>
                <tr className={`code-row${p.name === best ? " best" : ""}${p.ok ? "" : " muted"}`}
                    onClick={() => setOpen(open === p.name ? null : p.name)}>
                  <td className="mono">{p.name}</td>
                  <td className="num">{p.size.loc}</td>
                  {p.ok ? <>
                    <td className="num">{p.structure!.complexity}</td>
                    <td className="num">{p.structure!.for + p.structure!.while}</td>
                    <td className="num">{p.structure!.if}</td>
                    <td className="num">{p.structure!.functions}</td>
                    <td className="mono">{names(p, "model").join(", ")}</td>
                    <td className="mono muted">{names(p, "transformer").join(", ")}</td>
                    <td className="mono muted">{names(p, "splitter").join(", ")}</td>
                    <td className="mono muted">{p.metrics!.join(", ")}</td>
                    <td className="num">{p.data!.column_writes}</td>
                    <td className="num">{p.data!.reads.length}</td>
                  </> : <td colSpan={10} className="error">does not parse: {p.error}</td>}
                </tr>
                {open === p.name && <tr className="code-open"><td colSpan={12}><Detail p={p} /></td></tr>}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
