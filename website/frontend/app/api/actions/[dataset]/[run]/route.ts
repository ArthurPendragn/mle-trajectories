import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { getActions } from "@/lib/dal";
import { SESSION_COOKIE, decrypt } from "@/lib/session";

// Polled by the actions panel while a job runs. Read-only.

export async function GET(_: Request, ctx: RouteContext<"/api/actions/[dataset]/[run]">) {
  if (!(await decrypt((await cookies()).get(SESSION_COOKIE)?.value))) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const { dataset, run } = await ctx.params;
  const info = await getActions(dataset, run);
  if (!info) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(info, { headers: { "Cache-Control": "no-store" } });
}
