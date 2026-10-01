import "server-only";
import { cache } from "react";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { request } from "node:http";
import { homedir } from "node:os";
import { existsSync } from "node:fs";
import path from "node:path";
import { SESSION_COOKIE, decrypt } from "./session";
import type { ActionsInfo, AnalysisStatus, CodeAnalysis, Corpus, Job, RunDetail, RuntimeProfile, SampleParams,
  JobList, SamplePlan, SweepParams, SweepPlan, TreeData } from "./types";

// Data access layer: the only place that talks to the Python API, and every
// call re-checks the session (proxy.ts is only the optimistic first gate).

export const verifySession = cache(async () => {
  const session = await decrypt((await cookies()).get(SESSION_COOKIE)?.value);
  if (!session) redirect("/login");
  return session;
});

// Same resolution as website/backend/__main__.py.
function socketPath(): string {
  if (process.env.MLE_API_SOCKET) return process.env.MLE_API_SOCKET;
  const xdg = process.env.XDG_RUNTIME_DIR;
  const root = xdg && existsSync(xdg) ? xdg : path.join(homedir(), ".cache");
  return path.join(root, "mle-trajectories", "api.sock");
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function apiRequest<T>(method: "GET" | "POST", route: string, body?: unknown): Promise<T> {
  const payload = body === undefined ? undefined : JSON.stringify(body);
  return new Promise((resolve, reject) => {
    const req = request({
      socketPath: socketPath(), path: route, method,
      headers: payload === undefined ? {} : {
        "Content-Type": "application/json", "Content-Length": Buffer.byteLength(payload),
      },
    }, (res) => {
      let text = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => (text += chunk));
      res.on("end", () => {
        if (res.statusCode !== 200) {
          let detail = text;
          try {
            detail = JSON.parse(text).detail ?? text;
          } catch { /* not JSON */ }
          reject(new ApiError(res.statusCode ?? 500, String(detail)));
          return;
        }
        try {
          resolve(JSON.parse(text) as T);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on("error", (err) =>
      reject(new ApiError(503, `backend unreachable at ${socketPath()}: ${err.message}`)));
    req.end(payload);
  });
}

const apiGet = <T,>(route: string) => apiRequest<T>("GET", route);

export async function getCorpus(): Promise<Corpus> {
  await verifySession();
  return apiGet<Corpus>("/api/corpus");
}

function runPath(dataset: string, run: string): string {
  return `/api/runs/${encodeURIComponent(dataset)}/${encodeURIComponent(run)}`;
}

function query(params: Record<string, string | undefined | boolean>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== false) q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

async function orNull<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

export async function getRun(dataset: string, run: string,
                             sel: { source?: string; runtime?: string } = {}): Promise<RunDetail | null> {
  await verifySession();
  return orNull(apiGet<RunDetail>(runPath(dataset, run) + query(sel)));
}

export async function getTree(dataset: string, run: string, source?: string): Promise<TreeData> {
  await verifySession();
  return apiGet<TreeData>(`${runPath(dataset, run)}/tree${query({ source })}`);
}

export async function getCode(dataset: string, run: string): Promise<CodeAnalysis | null> {
  await verifySession();
  return orNull(apiGet<CodeAnalysis>(`${runPath(dataset, run)}/code`));
}

export async function getRuntimeProfile(dataset: string, run: string,
                                        store: string): Promise<RuntimeProfile | null> {
  await verifySession();
  return orNull(apiGet<RuntimeProfile>(
    `${runPath(dataset, run)}/runtime/${encodeURIComponent(store)}`));
}

/** Status of the operator analysis (its data when ready). ``start`` builds it in
 *  the background when missing, ``retry`` also after a failure. */
export async function getAnalysis(dataset: string, run: string, source: string,
                                  opts: { start?: boolean; retry?: boolean } = {}): Promise<AnalysisStatus | null> {
  await verifySession();
  return orNull(apiGet<AnalysisStatus>(
    `${runPath(dataset, run)}/analysis/${encodeURIComponent(source)}${query(opts)}`));
}

// --- actions -------------------------------------------------------------- //
// Mutations go through server actions (app/actions/), which get Next's Origin
// check; the API validates every parameter and builds the command itself.

export async function getActions(dataset: string, run: string): Promise<ActionsInfo | null> {
  await verifySession();
  return orNull(apiGet<ActionsInfo>(`${runPath(dataset, run)}/actions`));
}

export async function planSweep(dataset: string, run: string, params: SweepParams,
                                listing = false): Promise<SweepPlan> {
  await verifySession();
  return apiRequest<SweepPlan>("POST",
    `${runPath(dataset, run)}/actions/runtime-sweep/plan${query({ listing })}`, params);
}

export async function startSweep(dataset: string, run: string, params: SweepParams): Promise<Job> {
  await verifySession();
  return apiRequest<Job>("POST", `${runPath(dataset, run)}/actions/runtime-sweep`, params);
}

export async function planSample(dataset: string, run: string,
                                 params: SampleParams): Promise<SamplePlan> {
  await verifySession();
  return apiRequest<SamplePlan>("POST", `${runPath(dataset, run)}/actions/build-sample/plan`, params);
}

export async function startSample(dataset: string, run: string, params: SampleParams): Promise<Job> {
  await verifySession();
  return apiRequest<Job>("POST", `${runPath(dataset, run)}/actions/build-sample`, params);
}

export async function getJobs(limit = 100): Promise<JobList> {
  await verifySession();
  return apiGet<JobList>(`/api/jobs${query({ limit: String(limit) })}`);
}

export async function stopJob(id: string): Promise<Job> {
  await verifySession();
  return apiRequest<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/stop`);
}
