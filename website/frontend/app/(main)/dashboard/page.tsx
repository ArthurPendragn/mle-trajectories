import { getDashboard } from "@/lib/dal";
import { DashboardView } from "@/components/dashboard";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const data = await getDashboard();
  return (
    <>
      <h1>Global dashboard</h1>
      <p className="muted small">
        Static code analysis (<span className="mono">tools/code_stats</span>) of every run&apos;s original
        scripts, across the corpus. What a script imports from its run&apos;s own modules (mle-claude&apos;s
        <span className="mono"> common.py</span>, <span className="mono">features.py</span>, …) is inlined before
        measuring, each definition once, so a short script over shared helpers counts the code it uses
        {data.skipped.length > 0 && <>. Not covered ({data.skipped.length} run{data.skipped.length > 1 ? "s" : ""}:{" "}
          <span title={data.skipped.map((s) => `${s.id}: ${s.reason}`).join("\n")} className="hint-inline">
            {data.skipped.map((s) => s.id).join(", ")}</span>)</>}.
      </p>
      <DashboardView data={data} />
    </>
  );
}
