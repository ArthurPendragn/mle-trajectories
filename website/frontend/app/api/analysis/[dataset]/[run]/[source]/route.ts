import { NextResponse, type NextRequest } from "next/server";
import { cookies } from "next/headers";
import { getAnalysis } from "@/lib/dal";
import { SESSION_COOKIE, decrypt } from "@/lib/session";

// Polled by the run page while an analysis builds. Each poll also starts the
// build if it is missing (a queued build starts once a slot frees up); ?retry=1
// restarts a failed one. Starting/retrying is idempotent and only
// ever recomputes a cache, and the session cookie is SameSite=Strict, so a
// cross-site request cannot trigger it with the user's session.

export async function GET(req: NextRequest, ctx: RouteContext<"/api/analysis/[dataset]/[run]/[source]">) {
  if (!(await decrypt((await cookies()).get(SESSION_COOKIE)?.value))) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const { dataset, run, source } = await ctx.params;
  const retry = req.nextUrl.searchParams.get("retry") === "1";
  const status = await getAnalysis(dataset, run, source, { start: true, retry });
  if (!status) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(status, { headers: { "Cache-Control": "no-store" } });
}
