import "server-only";
import { cache } from "react";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { request } from "node:http";
import { homedir } from "node:os";
import { existsSync } from "node:fs";
import path from "node:path";
import { SESSION_COOKIE, decrypt } from "./session";
import type { AnalysisStatus, Corpus, RunDetail, RuntimeProfile, TreeData } from "./types";

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

function apiGet<T>(route: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const req = request({ socketPath: socketPath(), path: route, method: "GET" }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => {
        if (res.statusCode !== 200) {
          reject(new ApiError(res.statusCode ?? 500, body));
          return;
        }
        try {
          resolve(JSON.parse(body) as T);
        } catch (err) {
          reject(err);
        }
      });
    });
    req.on("error", (err) =>
      reject(new ApiError(503, `backend unreachable at ${socketPath()}: ${err.message}`)));
    req.end();
  });
}

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
