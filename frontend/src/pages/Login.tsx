import { Boxes, Eye, EyeOff, FileSearch, FileText } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";

import { useLogin, useMe, useRegister } from "../api/hooks";
import { Logo } from "../components/Layout";
import { Button, ErrorBox } from "../components/ui";

const FEATURES = [
  { icon: FileSearch, text: "Discover tenders that fit your catalogue" },
  { icon: Boxes, text: "AI matches requirements to your inventory" },
  { icon: FileText, text: "Price with GST and margin, export a polished PDF" },
];

export default function Login() {
  const { data: user } = useMe();
  const login = useLogin();
  const register = useRegister();
  const navigate = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from ?? "/";

  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [showPassword, setShowPassword] = useState(false);

  if (user) return <Navigate to={from} replace />;

  const error = (mode === "login" ? login.error : register.error ?? login.error) as Error | null;
  const busy = login.isPending || register.isPending;

  async function submit(e: FormEvent) {
    e.preventDefault();
    try {
      if (mode === "register") await register.mutateAsync({ email, name, password });
      await login.mutateAsync({ email, password, remember });
      navigate(from, { replace: true });
    } catch {
      // Shown in the form via login.error / register.error.
    }
  }

  return (
    <div className="grid h-full lg:grid-cols-2">
      <div className="relative hidden overflow-hidden bg-brand-50 p-12 lg:flex lg:flex-col">
        <div className="absolute -right-24 -top-24 h-72 w-72 rounded-full bg-brand-100/70" />
        <div className="absolute -bottom-24 -left-20 h-72 w-72 rotate-12 rounded-[3rem] bg-brand-100/60" />
        <Logo />
        <div className="relative my-auto max-w-md">
          <h1 className="text-4xl font-semibold leading-tight tracking-tight text-slate-800">
            Automate Your RFP Response Workflow
          </h1>
          <p className="mt-4 text-slate-600">Discover, match, price and generate RFP proposals faster with AI.</p>
          <ul className="mt-8 space-y-3">
            {FEATURES.map(({ icon: Icon, text }) => (
              <li key={text} className="flex items-center gap-3 text-sm text-slate-700">
                <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-brand-100 bg-white text-brand-500">
                  <Icon className="h-4 w-4" />
                </span>
                {text}
              </li>
            ))}
          </ul>
        </div>
        <p className="relative text-xs text-slate-500">© 2026 Synapse · BML Munjal University</p>
      </div>

      <div className="flex items-center justify-center bg-white p-6">
        <form onSubmit={submit} className="card w-full max-w-sm p-7">
          <div className="mb-5 flex h-9 w-9 items-center justify-center rounded-lg bg-brand-500 text-white">
            <FileText className="h-4 w-4" />
          </div>
          <h2 className="text-xl font-semibold text-slate-800">{mode === "login" ? "Welcome back" : "Create your account"}</h2>
          <p className="mt-1 text-sm text-slate-500">{mode === "login" ? "Sign in to your account" : "It only takes a moment"}</p>

          <div className="mt-6 space-y-4">
            {mode === "register" && (
              <div>
                <label className="label" htmlFor="name">Full name</label>
                <input id="name" className="input" value={name} onChange={(e) => setName(e.target.value)} required autoComplete="name" />
              </div>
            )}
            <div>
              <label className="label" htmlFor="email">Email</label>
              <input id="email" type="email" className="input" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
            </div>
            <div>
              <label className="label" htmlFor="password">Password</label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  className="input pr-10"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={mode === "register" ? 8 : undefined}
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-1 text-slate-400 hover:text-slate-600"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {mode === "register" && <p className="mt-1 text-xs text-slate-400">At least 8 characters.</p>}
            </div>
            <label className="flex items-center gap-2 text-sm text-slate-600">
              <input type="checkbox" className="h-4 w-4 rounded accent-brand-500" checked={remember} onChange={(e) => setRemember(e.target.checked)} />
              Remember me
            </label>
            {error && <ErrorBox title={mode === "login" ? "Could not sign in" : "Could not create the account"}>{error.message}</ErrorBox>}
            <Button type="submit" loading={busy} className="w-full">
              {mode === "login" ? "Sign In" : "Create account"}
            </Button>
          </div>

          <p className="mt-5 text-center text-sm text-slate-500">
            {mode === "login" ? "New here? " : "Already have an account? "}
            <button
              type="button"
              className="font-medium text-brand-600 hover:underline"
              onClick={() => {
                login.reset();
                register.reset();
                setMode(mode === "login" ? "register" : "login");
              }}
            >
              {mode === "login" ? "Create an account" : "Sign in"}
            </button>
          </p>
        </form>
      </div>
    </div>
  );
}
