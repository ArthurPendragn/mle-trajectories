import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, decrypt } from "@/lib/session";

// Optimistic gate: no valid session cookie -> /login. The real check is
// verifySession() in lib/dal.ts, next to every data access.
export default async function proxy(req: NextRequest) {
  const isLogin = req.nextUrl.pathname === "/login";
  const session = await decrypt(req.cookies.get(SESSION_COOKIE)?.value);
  if (!session && !isLogin) {
    return NextResponse.redirect(new URL("/login", req.nextUrl));
  }
  if (session && isLogin) {
    return NextResponse.redirect(new URL("/", req.nextUrl));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
