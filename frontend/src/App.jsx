import React from "react";
import { NavLink, Route, HashRouter, Routes } from "react-router-dom";

import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import AnalysisResult from "./pages/AnalysisResult";
import History from "./pages/History";
import DocumentDetail from "./pages/DocumentDetail";

export default function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <aside className="sidebar">
          <h1>DocuMind</h1>
          <nav>
            <NavLink to="/" end className={({ isActive }) => (isActive ? "active" : "")}>
              Dashboard
            </NavLink>
            <NavLink to="/upload" className={({ isActive }) => (isActive ? "active" : "")}>
              Upload
            </NavLink>
            <NavLink to="/history" className={({ isActive }) => (isActive ? "active" : "")}>
              History
            </NavLink>
          </nav>
        </aside>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/results/:id" element={<AnalysisResult />} />
            <Route path="/history" element={<History />} />
            <Route path="/documents/:id" element={<DocumentDetail />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}
