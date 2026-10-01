import Link from "next/link";
import { Suspense } from "react";
import { notFound } from "next/navigation";
import { getActions, getAnalysis, getCode, getRun, getRuntimeProfile, getTree } from "@/lib/dal";
import { capabilities, noPlansReason } from "@/lib/capabilities";
import { dataLabel, firstLine, fmtScore, improvement, metricLabel } from "@/lib/format";
import type { CodeDiff, RunDetail, Source } from "@/lib/types";
import { CodeAnalysisView } from "@/components/code-analysis";
import { RunAnalysis } from "@/components/run-analysis";
import { RuntimeProfile } from "@/components/runtime-profile";
import { ActionsPanel } from "@/components/actions/panel";
import { SelectionBar } from "@/components/selection-bar";
import { StaleHint } from "@/components/stale-hint";

export const dynamic = "force-dynamic";

function one(v: string | string[] | undefined): string | undefined {
  return Array.isArray(v) ? v[0] : v;
}

function fmtValue(v: unknown): string {
  return typeof v === "string" && v.startsWith("=") ? v.slice(1) : JSON.stringify(v);
}

/** The full step-vs-parent comparison, for the tooltip. */
function diffTitle(d: CodeDiff): string {
  const lines = [`vs ${d.parent}: ${Math.round(d.similarity * 100)}% similar, +${d.lines_added}/−${d.lines_removed} lines (loc ${d.loc_delta >= 0 ? "+" : ""}${d.loc_delta})`];
  if (d.components_added.length) lines.push(`added: ${d.components_added.join(", ")}`);
  if (d.components_removed.length) lines.push(`removed: ${d.components_removed.join(", ")}`);
  for (const p of d.params_changed) lines.push(`${p.component}.${p.param}: ${fmtValue(p.old)} → ${fmtValue(p.new)}`);
  if (d.imports_added.length) lines.push(`imports +${d.imports_added.join(", +")}`);
  if (d.imports_removed.length) lines.push(`imports −${d.imports_removed.join(", −")}`);
  if (d.reads_added.length) lines.push(`reads +${d.reads_added.join(", +")}`);
  if (d.reads_removed.length) lines.push(`reads −${d.reads_removed.join(", −")}`);
  return lines.join("\n");
}

function SourceRow({ s, run }: { s: Source; run: RunDetail }) {
  const [have, total] = s.coverage;
  const selected = s.name === run.selection.source?.name;
  return (
    <tr className={s.hidden ? "muted" : ""}>
      <td>
        {s.name}
        {selected && <span className="badge">selected</span>}
        {s.label && <div className="muted small">{s.label}</div>}
      </td>
      <td className="num"><span className={have === total ? "" : "partial"}>{have}/{total}</span></td>
      <td className="small">{s.dirs.join(", ")}</td>
      <td className="small">
        {s.fold_identical_code && <div>fold identical code</div>}
        {s.note}
        {s === run.originals && !s.skrub && <span className="muted">agent&apos;s original scripts</span>}
      </td>
    </tr>
  );
}

function Steps({ run, diffs }: { run: RunDetail; diffs: Record<string, CodeDiff> }) {
  const hasDiffs = Object.keys(diffs).length > 0;
  const score = new Map(run.steps.filter((s) => s.module).map((s) => [s.module!, s.score]));
  const best = run.best?.module;
  return (
    <table>
      <thead>
        <tr>
          <th>#</th><th>Pipeline</th><th>Phase</th><th>Parent</th>
          <th className="num">Score</th><th className="num">Δ parent</th>
          {hasDiffs && <><th className="num" title="added + removed lines against the parent">Lines changed</th>
            <th className="num" title="changed code lines (blank and comment lines ignored) as a share of the parent's code lines">Changed</th>
            <th>Code vs parent</th></>}<th>Description</th>
        </tr>
      </thead>
      <tbody>
        {run.steps.map((s, i) => {
          const d = improvement(s.score, s.parent ? score.get(s.parent) ?? null : null, run.metric);
          return (
            <tr key={i} id={s.module ? `step-${s.module}` : undefined}
                className={s.module === best ? "best" : s.module ? "" : "muted"}>
              <td className="num muted">{i + 1}</td>
              <td className="mono small">{s.module ?? "(no code kept)"}</td>
              <td>{s.phase ?? ""}</td>
              <td className="mono small muted">{s.parent ?? ""}</td>
              <td className="num">{fmtScore(s.score)}</td>
              <td className={`num ${d === null ? "" : d > 0 ? "up" : d < 0 ? "down" : "muted"}`}>
                {d === null ? "" : `${d > 0 ? "+" : ""}${fmtScore(d)}`}
              </td>
              {hasDiffs && (() => {
                const d = s.module ? diffs[s.module] : undefined;
                return <>
                  <td className="num small">
                    {d && (d.same_code ? <span className="muted">0</span>
                      : <span title={`+${d.lines_added} / −${d.lines_removed}`}>{d.lines_changed}</span>)}
                  </td>
                  <td className="num small muted">{d?.change_ratio != null ? `${Math.round(d.change_ratio * 100)}%` : ""}</td>
                  <td className="small code-delta" title={d ? diffTitle(d) : undefined}>{d?.summary ?? ""}</td>
                </>;
              })()}
              <td className="small" title={s.desc ?? undefined}>{firstLine(s.desc)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default async function RunPage({ params, searchParams }: PageProps<"/runs/[dataset]/[run]">) {
  const { dataset, run: name } = await params;
  const q = await searchParams;
  const run = await getRun(dataset, name, { source: one(q.source), runtime: one(q.runtime) });
  if (!run) notFound();
  const ds = run.dataset_info;
  const src = run.selection.source?.name ?? null;
  const rtName = run.selection.runtime?.name ?? null;
  const noPlans = noPlansReason(run);

  // the operator analysis is started here (idempotent) so it is building by the
  // time the page is on screen; the tree and runtime profile need no build
  const [tree, analysis, profile, actionsInfo, code] = await Promise.all([
    getTree(dataset, name, src ?? undefined),
    src && !noPlans ? getAnalysis(dataset, name, src, { start: true }) : Promise.resolve(null),
    rtName ? getRuntimeProfile(dataset, name, rtName) : Promise.resolve(null),
    getActions(dataset, name),
    getCode(dataset, name),
  ]);
  const store = run.runtime.find((r) => r.name === rtName) ?? null;
  const actions = capabilities(run);
  const sweep = actions.find((a) => a.key === "sweep");

  return (
    <>
      <p className="crumbs"><Link href="/">Corpus</Link> / {ds.label}</p>
      <h1>
        {run.label} <span className={`badge agent-${run.agent}`}>{run.agent ?? "agent ?"}</span>
      </h1>
      <p className="muted mono small">{run.path}</p>
      {run.note && <p className="note">{run.note}</p>}

      <div className="facts">
        <div><span className="muted">metric</span>{metricLabel(run.metric)}</div>
        <div><span className="muted">lineage</span>
          {run.lineage ? `${run.lineage}${run.trajectory ? ` (${run.trajectory.file})` : ""}` : "none"}</div>
        <div><span className="muted">steps</span>{run.n_steps}</div>
        <div><span className="muted">best</span>
          {fmtScore(run.best?.score)} <span className="mono small muted">{run.best?.module}</span></div>
        <div><span className="muted">data</span>
          {ds.data_status}{ds.samples.length > 0 && ` · samples: ${ds.samples.join(", ")}`}</div>
        <div><span className="muted">actions</span>
          <span>
            {actions.map((a) => a.ok
              ? <a key={a.key} className="action" href="#actions">{a.label}</a>
              : <button key={a.key} className="action" disabled title={a.reason}>{a.label}</button>)}
          </span>
        </div>
      </div>

      <Suspense>
        <SelectionBar sources={run.sources} runtime={run.runtime} selection={run.selection} />
      </Suspense>

      <RunAnalysis dataset={dataset} run={name} source={noPlans ? null : src}
                   noSourceReason={noPlans ?? ""} tree={tree} initial={analysis}
                   afterTree={code && (
                     <section key="code" id="code">
                       <h2>Code</h2>
                       <CodeAnalysisView data={code} best={run.best?.module ?? null}
                         scores={Object.fromEntries(run.steps.filter((s) => s.module).map((s) => [s.module!, s.score]))} />
                     </section>
                   )} />

      <section id="actions">
        <h2>Actions</h2>
        {!actionsInfo ? <p className="muted">Unavailable.</p>
          : <ActionsPanel dataset={dataset} run={name} initial={actionsInfo} selectedSource={src}
                          stores={run.runtime.map((r) => r.name)}
                          sweepReason={sweep?.ok ? null : sweep?.reason ?? "unknown run"} />}
      </section>

      <section id="runtime-profile">
        <h2>Runtime profile</h2>
        {!profile || !store
          ? <p className="muted">No runtime measurements for this run yet.</p>
          : <RuntimeProfile profile={profile} store={store} current={run.stratum_commit} />}
      </section>

      <section>
        <h2>Pipeline sources</h2>
        <table>
          <thead><tr><th>Source</th><th className="num">Coverage</th><th>Folders</th><th /></tr></thead>
          <tbody>
            {run.originals && !run.originals.skrub && <SourceRow s={run.originals} run={run} />}
            {run.sources.map((s) => <SourceRow key={s.name} s={s} run={run} />)}
          </tbody>
        </table>
      </section>

      <section>
        <h2>Runtime stores</h2>
        {run.runtime.length === 0 ? <p className="muted">None yet.</p> : (
          <table>
            <thead>
              <tr><th>Store</th><th>Source</th><th className="num">OK</th>
                <th className="num">Failed</th><th>Data</th><th /></tr>
            </thead>
            <tbody>
              {run.runtime.map((r) => (
                <tr key={r.file} className={r.hidden ? "muted" : ""}>
                  <td className="mono small">
                    {r.file}{r.name === rtName && <span className="badge">selected</span>}
                  </td>
                  <td>{r.source ?? <span className="muted">?</span>}</td>
                  <td className="num">{r.n_ok}</td>
                  <td className="num">{r.n_failed || ""}</td>
                  <td>
                    {dataLabel(r)}
                  </td>
                  <td><StaleHint store={r} current={run.stratum_commit} compact /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {run.trajectory && (
        <section>
          <h2>Trajectory</h2>
          <dl className="meta">
            {Object.entries(run.trajectory.meta).map(([k, v]) => (
              <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
            ))}
          </dl>
        </section>
      )}

      <section>
        <h2>Steps</h2>
        {run.steps.length === 0 ? <p className="muted">No lineage recorded for this run.</p>
          : <Steps run={run} diffs={code?.diffs ?? {}} />}
      </section>

      {(run.warnings.length > 0 || ds.warnings.length > 0) && (
        <section>
          <h2>Warnings</h2>
          <ul>{[...ds.warnings, ...run.warnings].map((w) => <li key={w}>{w}</li>)}</ul>
        </section>
      )}
    </>
  );
}
