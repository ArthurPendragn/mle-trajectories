import Link from "next/link";
import { logout } from "@/app/actions/auth";
import { verifySession } from "@/lib/dal";

export default async function MainLayout({ children }: { children: React.ReactNode }) {
  const session = await verifySession();
  return (
    <>
      <header className="topbar">
        <Link href="/" className="brand">MLE trajectories</Link>
        <nav>
          <Link href="/">Corpus</Link>
        </nav>
        <form action={logout} className="logout">
          <span className="muted">{session.user}</span>
          <button type="submit" className="link-button">Sign out</button>
        </form>
      </header>
      <main className="page">{children}</main>
    </>
  );
}
