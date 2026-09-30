import Link from "next/link";
import { getJobs } from "@/lib/dal";
import { JobsOverview } from "@/components/actions/jobs-overview";

export const dynamic = "force-dynamic";

export default async function JobsPage() {
  const list = await getJobs();
  return (
    <>
      <p className="crumbs"><Link href="/">Corpus</Link> / Jobs</p>
      <h1>Jobs</h1>
      <p className="muted">
        Every sample build and runtime sweep started from the website, newest first. Start new
        ones from a run page&apos;s Actions section. State lives in{" "}
        <span className="mono">website/.cache/actions/</span>.
      </p>
      <JobsOverview initial={list} />
    </>
  );
}
