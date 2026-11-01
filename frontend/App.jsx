import React, { useEffect, useState } from "react";
import * as api from "./api.js";
import { Archive, ClaimStudio, EvidenceChamber, ContextGraph, StanceObservatory, NeuralLab, ResearchJournal } from "./screens.jsx";

const TABS = [
  ["archive", "The Archive"],
  ["studio", "Claim Studio"],
  ["chamber", "Evidence Chamber"],
  ["graph", "Context Graph"],
  ["stance", "Stance Observatory"],
  ["lab", "Neural Lab"],
  ["journal", "Research Journal"],
];

export default function App() {
  const [tab, setTab] = useState("archive");
  const [status, setStatus] = useState(null);
  const [result, setResult] = useState(null);
  const [selected, setSelected] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.health().then(setStatus).catch(() => setStatus({ model_ready: false, offline: true }));
  }, []);

  async function run(text, evidence) {
    setBusy(true);
    setError("");
    try {
      const out = await api.analyze(text, evidence);
      setResult(out);
      setSelected(0);
      setTab(out.claims.length ? "chamber" : "studio");
      if (!out.claims.length) setError("No checkable claims found. Add sentences that state a fact, a number or an event.");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const shared = { result, selected, setSelected, go: setTab };
  return (
    <div className="shell">
      <header className="masthead">
        <button className="wordmark" onClick={() => setTab("archive")}>Veritas</button>
        <nav aria-label="Screens">
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? "tab on" : "tab"} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
      </header>
      <main>
        {tab === "archive" && <Archive status={status} go={setTab} />}
        {tab === "studio" && <ClaimStudio run={run} busy={busy} error={error} status={status} />}
        {tab === "chamber" && <EvidenceChamber {...shared} />}
        {tab === "graph" && <ContextGraph {...shared} />}
        {tab === "stance" && <StanceObservatory {...shared} />}
        {tab === "lab" && <NeuralLab />}
        {tab === "journal" && <ResearchJournal />}
      </main>
    </div>
  );
}
