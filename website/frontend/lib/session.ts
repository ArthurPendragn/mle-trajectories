import "server-only";
import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";

// Stateless session: a signed (HS256) cookie holding only the user name and
// expiry. Changing SESSION_SECRET (set-password does) logs every session out.

export const SESSION_COOKIE = "mle_session";
const MAX_AGE_S = 7 * 24 * 60 * 60;

function key(): Uint8Array {
  const secret = process.env.SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error("SESSION_SECRET is not set -- run `npm run set-password`");
  }
  return new TextEncoder().encode(secret);
}

export async function decrypt(token: string | undefined): Promise<{ user: string } | null> {
  if (!token) return null;
  try {
    const { payload } = await jwtVerify(token, key(), { algorithms: ["HS256"] });
    return typeof payload.user === "string" ? { user: payload.user } : null;
  } catch {
    return null;
  }
}

export async function createSession(user: string): Promise<void> {
  const token = await new SignJWT({ user })
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setExpirationTime(`${MAX_AGE_S}s`)
    .sign(key());
  (await cookies()).set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: "strict",
    // the site is reached as http://localhost through an SSH tunnel, so the
    // cookie never crosses the network in clear; set COOKIE_SECURE=1 behind https
    secure: process.env.COOKIE_SECURE === "1",
    maxAge: MAX_AGE_S,
    path: "/",
  });
}

export async function deleteSession(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
}
