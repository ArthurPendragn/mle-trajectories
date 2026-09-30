"use server";

import { ApiError, planSweep, startSweep, stopJob } from "@/lib/dal";
import type { Job, SweepParams, SweepPlan } from "@/lib/types";

// Server actions for the run page's actions panel. Next checks each call's
// Origin against the host (no cross-site trigger); the DAL re-checks the
// session; the API validates the parameters and builds the command.

export type Result<T> = { ok: true; value: T } | { ok: false; error: string };

async function wrap<T>(p: Promise<T>): Promise<Result<T>> {
  try {
    return { ok: true, value: await p };
  } catch (err) {
    if (err instanceof ApiError) return { ok: false, error: err.message };
    throw err;
  }
}

export async function planSweepAction(dataset: string, run: string, params: SweepParams,
                                      listing: boolean): Promise<Result<SweepPlan>> {
  return wrap(planSweep(dataset, run, params, listing));
}

export async function startSweepAction(dataset: string, run: string,
                                       params: SweepParams): Promise<Result<Job>> {
  return wrap(startSweep(dataset, run, params));
}

export async function stopJobAction(id: string): Promise<Result<Job>> {
  return wrap(stopJob(id));
}
