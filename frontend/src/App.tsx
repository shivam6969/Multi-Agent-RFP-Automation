import type { ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import { useMe } from "./api/hooks";
import Layout from "./components/Layout";
import { Spinner } from "./components/ui";
import History from "./pages/History";
import Home from "./pages/Home";
import Login from "./pages/Login";
import NewSession from "./pages/NewSession";
import ProjectPage from "./pages/ProjectPage";
import Projects from "./pages/Projects";
import RfpSelection from "./pages/RfpSelection";
import Settings from "./pages/Settings";

function RequireLogin({ children }: { children: ReactNode }) {
  const { data: user, isLoading } = useMe();
  const location = useLocation();
  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        element={
          <RequireLogin>
            <Layout />
          </RequireLogin>
        }
      >
        <Route index element={<Home />} />
        <Route path="new" element={<NewSession />} />
        <Route path="rfps" element={<RfpSelection />} />
        <Route path="projects" element={<Projects />} />
        <Route path="projects/:projectId" element={<ProjectPage />} />
        <Route path="history" element={<History />} />
        <Route path="settings" element={<Settings />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
