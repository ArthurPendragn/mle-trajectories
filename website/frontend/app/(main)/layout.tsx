import Link from "next/link";
import { logout } from "@/app/actions/auth";
import { verifySession } from "@/lib/dal";
import { ThemeToggle } from "@/components/theme-toggle";

export default async function MainLayout({ children }: { children: React.ReactNode }) {
  const session = await verifySession();
  return (
    <>
      <header className="topbar">
        <Link href="/" className="brand">MLE trajectories</Link>
        <nav>
          <Link href="/">Corpus</Link>
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/jobs">Jobs</Link>
        </nav>
        <form action={logout} className="logout">
          <ThemeToggle />
          <span className="muted">{session.user}</span>
          <button type="submit" className="link-button">Sign out</button>
        </form>
      </header>
      <main className="page">{children}</main>
    </>
  );
}
