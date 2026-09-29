import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MLE trajectories",
  description: "Search trajectories of ML engineering agents, as skrub DataOps plans",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
