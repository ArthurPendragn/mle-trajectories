import "server-only";
import { scrypt, timingSafeEqual } from "node:crypto";

// One account, configured in .env.local by `npm run set-password`:
//   AUTH_USER=<name>
//   AUTH_PASSWORD_HASH=scrypt:<N>:<r>:<p>:<salt b64url>:<hash b64url>
// (':' separated on purpose: .env files expand `$VAR`, which would mangle a
// `$`-separated hash.)

function scryptAsync(pw: string, salt: Buffer, len: number, N: number, r: number, p: number) {
  return new Promise<Buffer>((resolve, reject) =>
    scrypt(pw, salt, len, { N, r, p, maxmem: 256 * N * r }, (err, key) =>
      err ? reject(err) : resolve(key),
    ),
  );
}

async function checkPassword(password: string, stored: string): Promise<boolean> {
  const [kind, N, r, p, salt, hash] = stored.split(":");
  if (kind !== "scrypt" || !hash) return false;
  const expected = Buffer.from(hash, "base64url");
  const actual = await scryptAsync(password, Buffer.from(salt, "base64url"),
    expected.length, Number(N), Number(r), Number(p));
  return timingSafeEqual(actual, expected);
}

// Throttle: behind the tunnel every request comes from 127.0.0.1, so the
// counter is global rather than per client. After MAX_FAILURES failed attempts
// within WINDOW_MS, logins are refused for LOCK_MS; every failure also waits a
// second before answering.
const MAX_FAILURES = 5;
const WINDOW_MS = 15 * 60_000;
const LOCK_MS = 5 * 60_000;
let failures: number[] = [];
let lockedUntil = 0;

export type LoginResult = "ok" | "invalid" | "locked" | "unconfigured";

export async function verifyCredentials(user: string, password: string): Promise<LoginResult> {
  const now = Date.now();
  if (now < lockedUntil) return "locked";
  const expectedUser = process.env.AUTH_USER;
  const stored = process.env.AUTH_PASSWORD_HASH;
  if (!expectedUser || !stored) return "unconfigured";

  const given = Buffer.from(user);
  const wanted = Buffer.from(expectedUser);
  const userOk = given.length === wanted.length && timingSafeEqual(given, wanted);
  // always hash, so a wrong user name costs as long as a wrong password
  const passOk = await checkPassword(password, stored).catch(() => false);
  if (userOk && passOk) {
    failures = [];
    return "ok";
  }
  failures = [...failures.filter((t) => now - t < WINDOW_MS), now];
  if (failures.length >= MAX_FAILURES) {
    lockedUntil = now + LOCK_MS;
    failures = [];
  }
  await new Promise((r) => setTimeout(r, 1000));
  return "invalid";
}
