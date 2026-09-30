import { dataLabel, fmtScore, fmtSecs } from "@/lib/format";
import type { RuntimeProfile as Profile, RuntimeStoreDetail } from "@/lib/types";
import { StaleHint } from "./stale-hint";

/** Where the measured pipelines spent their time, from one runtime store. */
export function RuntimeProfile({ profile, store, current }: {
  profile: Profile;
  store: RuntimeStoreDetail;
  current: string | null;
}) {
  const where = profile.data === "input" && !profile.legacy_rows ? "the full data" : dataLabel(profile);
  return (
    <>
      <p className="muted small">
        {profile.n_measured} pipeline(s) measured on {where}
        {profile.n_failed > 0 && `, ${profile.n_failed} failed`}:{" "}
        {fmtSecs(profile.op_total_s)} inside operator bodies out of {fmtSecs(profile.wall_total_s)}{" "}
        of scored grid search. <b>Per call</b> separates an operator that is expensive from one
        that is merely frequent.
        {profile.legacy_rows !== null && " This store comes from the removed row cap, which wrapped read_csv: its read shows as CallOp and is not comparable. Re-measure it on a sample folder."}
      </p>
      <StaleHint store={store} current={current} />
      {store.note && <p className="muted small">{store.note}</p>}

      <h3>Time per operator class</h3>
      <table>
        <thead>
          <tr>
            <th>operator</th><th className="num">total</th><th className="num">share</th>
            <th className="num">calls</th><th className="num">per call</th>
            <th className="num">pipelines</th><th>heaviest in</th>
          </tr>
        </thead>
        <tbody>
          {profile.ops.map((o) => (
            <tr key={o.op}>
              <td className="mono small">{o.op}</td>
              <td className="num">{fmtSecs(o.time_s)}</td>
              <td className="num">{o.share === null ? "—" : `${(100 * o.share).toFixed(1)}%`}</td>
              <td className="num">{o.calls}</td>
              <td className="num">{fmtSecs(o.per_call_s)}</td>
              <td className="num">{o.n_pipelines}/{profile.n_measured}</td>
              <td className="small">
                <a href={`#step-${o.heaviest.name}`} className="mono">{o.heaviest.name}</a>{" "}
                <span className="muted">{fmtSecs(o.heaviest.time_s)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <details className="block" open={profile.pipelines.length <= 12}>
        <summary>Per pipeline ({profile.pipelines.length}), slowest first</summary>
        <table>
          <thead>
            <tr>
              <th>pipeline</th><th className="num">wall</th><th className="num">operators</th>
              <th className="num">peak MB</th><th className="num">mean MB</th>
              <th className="num">op calls</th><th className="num">CV score</th><th />
            </tr>
          </thead>
          <tbody>
            {profile.pipelines.map((p) => (
              <tr key={p.name} className={p.status === "ok" ? "" : "muted"}>
                <td className="mono small"><a href={`#step-${p.name}`}>{p.name}</a></td>
                <td className="num">{fmtSecs(p.wall_s)}</td>
                <td className="num">{fmtSecs(p.op_time_s)}</td>
                <td className="num">{p.max_rss_mb === null ? "—" : Math.round(p.max_rss_mb)}</td>
                <td className="num">{p.mean_mb === null ? "—" : Math.round(p.mean_mb)}</td>
                <td className="num">{p.n_op_calls ?? "—"}</td>
                <td className="num">{fmtScore(p.best_score)}</td>
                <td className="small muted">
                  {p.status !== "ok" ? (p.error?.[0] ?? p.status)
                    : p.code_changed ? "file changed since" : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </>
  );
}
