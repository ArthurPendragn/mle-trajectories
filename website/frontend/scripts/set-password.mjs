#!/usr/bin/env node
// Configure the single login: writes AUTH_USER, AUTH_PASSWORD_HASH (scrypt) and a
// fresh SESSION_SECRET into .env.local (mode 0600), keeping any other lines.
// A new secret signs out every existing session.
//
//   npm run set-password            # prompts for user and password
import { randomBytes, scryptSync } from "node:crypto";
import { chmodSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ENV = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", ".env.local");
const N = 1 << 15, r = 8, p = 1;

function ask(question, hidden = false) {
  const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: true });
  if (hidden) {
    rl._writeToOutput = (s) => { if (s.includes(question)) rl.output.write(s); };
  }
  return new Promise((resolve) => rl.question(question, (answer) => {
    rl.close();
    if (hidden) process.stdout.write("\n");
    resolve(answer);
  }));
}

const user = (await ask("user: ")).trim();
if (!user) { console.error("empty user name"); process.exit(1); }
const pw = await ask("password: ", true);
if (pw.length < 12) { console.error("use at least 12 characters"); process.exit(1); }
if (pw !== await ask("repeat: ", true)) { console.error("passwords differ"); process.exit(1); }

const salt = randomBytes(16);
const hash = scryptSync(pw, salt, 32, { N, r, p, maxmem: 256 * N * r });
const values = {
  AUTH_USER: user,
  AUTH_PASSWORD_HASH: ["scrypt", N, r, p, salt.toString("base64url"), hash.toString("base64url")].join(":"),
  SESSION_SECRET: randomBytes(32).toString("base64url"),
};

const kept = existsSync(ENV)
  ? readFileSync(ENV, "utf8").split("\n").filter((l) => l && !(l.split("=")[0] in values))
  : [];
writeFileSync(ENV, [...kept, ...Object.entries(values).map(([k, v]) => `${k}=${v}`)].join("\n") + "\n",
  { mode: 0o600 });
chmodSync(ENV, 0o600);
console.log(`wrote ${ENV} -- restart the frontend to pick it up`);
