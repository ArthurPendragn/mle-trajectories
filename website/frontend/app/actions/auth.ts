"use server";

import { redirect } from "next/navigation";
import { verifyCredentials } from "@/lib/credentials";
import { createSession, deleteSession } from "@/lib/session";

export type LoginState = { error?: string } | undefined;

const MESSAGES = {
  invalid: "Wrong user name or password.",
  locked: "Too many failed attempts. Try again in a few minutes.",
  unconfigured: "No account configured. Run `npm run set-password` on the server.",
};

export async function login(_: LoginState, form: FormData): Promise<LoginState> {
  const user = String(form.get("user") ?? "").slice(0, 200);
  const password = String(form.get("password") ?? "").slice(0, 1000);
  const result = await verifyCredentials(user, password);
  if (result !== "ok") return { error: MESSAGES[result] };
  await createSession(user);
  redirect("/");
}

export async function logout(): Promise<void> {
  await deleteSession();
  redirect("/login");
}
