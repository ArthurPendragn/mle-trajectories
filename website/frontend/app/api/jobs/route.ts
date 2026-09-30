import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { getJobs } from "@/lib/dal";
import { SESSION_COOKIE, decrypt } from "@/lib/session";

// Polled by the jobs overview while a job runs. Read-only.

export async function GET() {
  if (!(await decrypt((await cookies()).get(SESSION_COOKIE)?.value))) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  return NextResponse.json(await getJobs(), { headers: { "Cache-Control": "no-store" } });
}
