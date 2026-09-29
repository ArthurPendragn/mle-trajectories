import type { Metadata } from "next";
import LoginForm from "./login-form";

export const metadata: Metadata = { title: "Sign in · MLE trajectories" };

export default function LoginPage() {
  return (
    <main className="login">
      <h1>MLE trajectories</h1>
      <LoginForm />
    </main>
  );
}
