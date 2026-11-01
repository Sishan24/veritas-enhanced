import React, { useEffect, useState } from "react";
import * as api from "./api.js";

export const VERDICT = {
  SUPPORTED: { label: "Supported", color: "var(--support)" },
  REFUTED: { label: "Refuted", color: "var(--refute)" },
  CONFLICTING: { label: "Conflicting evidence", color: "var(--conflict)" },
  INSUFFICIENT: { label: "Insufficient evidence", color: "var(--unsure)" },
};
const STANCE = { SUPPORTED: "var(--support)", REFUTED: "var(--refute)", NOT_ENOUGH_INFO: "var(--unsure)" };
const STANCE_LABEL = { SUPPORTED: "Supports", REFUTED: "Contradicts", NOT_ENOUGH_INFO: "Not enough to say" };
const pct = (x) => `${Math.round(x * 100)}%`;

const SAMPLE = `The United Nations Security Council held an emergency session on the Gaza ceasefire proposal. Spain won the UEFA Euro 2024 championship. The US Federal Reserve raised interest rates to a 22-year high. Global CO2 emissions from fossil fuels reached a new record in 2023. No credible evidence supports claims that 5G towers spread COVID-19. Bitcoin surpassed 70 000 dollars in price in 2024. Canada experienced its worst wildfire season on record in 2023.`;

function Empty({ go }) {
  return (
    <section className="page narrow">
      <h2>Nothing analysed yet</h2>
      <p>Paste an article or statement in Claim Studio and VERITAS will list each claim with the evidence behind it.</p>
      <button className="primary" onClick={() => go("studio")}>Open Claim Studio</button>
    </section>
  );
}

function VerdictTag({ v }) {
  return <span className="tag" style={{ "--c": VERDICT[v].color }}>{VERDICT[v].label}</span>;
}

function ClaimPicker({ result, selected, setSelected }) {
  return (
    <ol className="claims">
      {result.claims.map((c, i) => (
        <li key={c.id}>
          <button className={i === selected ? "claim on" : "claim"} onClick={() => setSelected(i)}>
            <span>{c.text}</span>
            <VerdictTag v={c.verdict} />
          </button>
        </li>
      ))}
    </ol>
  );
}

export function Archive({ status, go }) {
  return (
    <section className="page hero">
      <h1>What does this claim actually rest on?</h1>
      <p className="lede">
        Paste an article or statement. VERITAS splits it into claims, finds the passages that back or contradict each
        one, and shows which words in the evidence drove the verdict. A claim with no evidence stays unverified. It is never marked false by default.
      </p>
      <div className="verdicts">
        {Object.entries(VERDICT).map(([k, v]) => (
          <span key={k} className="tag" style={{ "--c": v.color }}>{v.label}</span>
        ))}
      </div>
      <button className="primary" onClick={() => go("studio")}>Open Claim Studio</button>
      <p className="status">
        {status === null && "Checking the analysis server."}
        {status?.offline && "The analysis server is not running. Start it with: python -m uvicorn backend.app.main:app --reload"}
        {status && !status.offline && !status.model_ready && "Server is running, but no model is trained yet. Run: python -m ml.make_sample_data, then python -m ml.train"}
        {status?.model_ready && `Ready. ${status.arch} model loaded, ${status.default_evidence} evidence passages indexed.`}
      </p>
    </section>
  );
}

export function ClaimStudio({ run, busy, error, status }) {
  const [text, setText] = useState(SAMPLE);
  const [evidence, setEvidence] = useState([]);

  function onFiles(e) {
    Promise.all([...e.target.files].map((f) => f.text().then((t) => ({ id: f.name, text: t })))).then((docs) =>
      setEvidence((prev) => [...prev, ...docs])
    );
  }

  return (
    <section className="page studio">
      <div>
        <label htmlFor="claims">Article or statement</label>
        <textarea id="claims" rows={10} value={text} onChange={(e) => setText(e.target.value)} />
      </div>
      <div>
        <label htmlFor="ev">Evidence documents</label>
        <input id="ev" type="file" accept=".txt,.md" multiple onChange={onFiles} />
        <p className="hint">
          {evidence.length
            ? `${evidence.length} file(s) added: ${evidence.map((d) => d.id).join(", ")}.`
            : "Optional. With no files, claims are checked against the built-in sample evidence."}
        </p>
        {evidence.length > 0 && <button className="quiet" onClick={() => setEvidence([])}>Remove files</button>}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      <button className="primary" disabled={busy || !text.trim() || status?.model_ready === false} onClick={() => run(text, evidence)}>
        {busy ? "Analysing" : "Analyse claims"}
      </button>
    </section>
  );
}

function Highlighted({ tokens }) {
  if (!tokens) return <span className="muted">This model does not expose attention weights. Use bilstm_attn for word-level evidence.</span>;
  const max = Math.max(...tokens.map((t) => t.weight), 1e-9);
  return (
    <p className="attn">
      {tokens.map((t, i) => (
        <mark key={i} style={{ "--a": (t.weight / max).toFixed(2) }} title={`weight ${t.weight.toFixed(3)}`}>{t.token} </mark>
      ))}
    </p>
  );
}

export function EvidenceChamber({ result, selected, setSelected, go }) {
  if (!result?.claims.length) return <Empty go={go} />;
  const claim = result.claims[selected];
  return (
    <section className="page chamber">
      <ClaimPicker {...{ result, selected, setSelected }} />
      <div>
        <h2>{claim.text}</h2>
        <p className="verdict-line">
          <VerdictTag v={claim.verdict} /> <span>{pct(claim.confidence)} confidence from {claim.evidence.length} retrieved passage(s)</span>
        </p>
        {claim.evidence.length === 0 && <p>No evidence passages were found for this claim.</p>}
        {claim.evidence.map((e) => (
          <article key={e.id} className="card" style={{ "--c": STANCE[e.stance] }}>
            <header>
              <strong>{STANCE_LABEL[e.stance]}</strong>
              <span className="muted">{pct(e.probs[e.stance])} · match to claim {e.retrieval_score.toFixed(2)} · {e.id}</span>
            </header>
            <p className="quote">{e.text}</p>
            <h4>Words the model weighed most</h4>
            <Highlighted tokens={e.attention} />
          </article>
        ))}
      </div>
    </section>
  );
}

export function ContextGraph({ result, selected, setSelected, go }) {
  if (!result?.claims.length) return <Empty go={go} />;
  const claim = result.claims[selected];
  const n = claim.evidence.length;
  const R = 150, cx = 260, cy = 210;
  return (
    <section className="page chamber">
      <ClaimPicker {...{ result, selected, setSelected }} />
      <figure>
        <svg viewBox="0 0 520 420" role="img" aria-label={`Evidence constellation for: ${claim.text}`}>
          {claim.evidence.map((e, i) => {
            const a = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2;
            const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
            const col = STANCE[e.stance];
            return (
              <g key={e.id}>
                <line x1={cx} y1={cy} x2={x} y2={y} stroke={col} strokeWidth={1 + 5 * e.probs[e.stance]}
                  strokeDasharray={e.stance === "NOT_ENOUGH_INFO" ? "5 5" : undefined} />
                <circle cx={x} cy={y} r={8 + 10 * e.retrieval_score} fill={col} />
                <text x={x} y={y + (y < cy ? -26 : 34)} textAnchor="middle" className="svg-label">{e.id}</text>
              </g>
            );
          })}
          <circle cx={cx} cy={cy} r="30" fill={VERDICT[claim.verdict].color} />
          <text x={cx} y={cy + 4} textAnchor="middle" className="svg-claim">claim</text>
        </svg>
        <figcaption>
          Line colour shows the stance, line weight the model's certainty, and node size how closely the passage matches the claim. Dashed lines mean the passage does not settle it.
        </figcaption>
      </figure>
    </section>
  );
}

export function StanceObservatory({ result, go }) {
  if (!result?.claims.length) return <Empty go={go} />;
  const counts = result.claims.reduce((a, c) => ({ ...a, [c.verdict]: (a[c.verdict] || 0) + 1 }), {});
  return (
    <section className="page narrow">
      <h2>{result.claims.length} claims, by verdict</h2>
      <div className="stack" role="img" aria-label="Verdict distribution">
        {Object.keys(VERDICT).filter((k) => counts[k]).map((k) => (
          <div key={k} style={{ flex: counts[k], background: VERDICT[k].color }}>{counts[k]} {VERDICT[k].label.toLowerCase()}</div>
        ))}
      </div>
      {result.claims.map((c) => (
        <div key={c.id} className="obs">
          <p>{c.text} <VerdictTag v={c.verdict} /></p>
          {c.evidence.map((e) => (
            <div key={e.id} className="bars" title={e.text}>
              <span className="muted">{e.id}</span>
              <div className="stack thin">
                {Object.keys(STANCE).map((k) => <div key={k} style={{ flex: e.probs[k], background: STANCE[k] }} />)}
              </div>
            </div>
          ))}
        </div>
      ))}
      <p className="hint">Bars show each passage's probability split: supports (green), contradicts (red), not enough to say (ochre).</p>
    </section>
  );
}

const ARCH_NOTES = [
  ["mlp", "Averages word vectors, then classifies. The baseline that ignores word order."],
  ["cnn", "Convolutions over 2 to 4 word windows capture local phrasing."],
  ["lstm", "Reads the text left to right and carries context forward."],
  ["bilstm", "Reads in both directions so each word sees what comes before and after."],
  ["bilstm_attn", "BiLSTM plus attention: weights evidence words by relevance to the claim. This is the model behind the highlighted words."],
];

export function NeuralLab() {
  const [rows, setRows] = useState(null);
  useEffect(() => { api.reports().then((r) => setRows(r.comparison)).catch(() => setRows([])); }, []);
  return (
    <section className="page narrow">
      <h2>Architectures compared</h2>
      <dl className="arch">{ARCH_NOTES.map(([k, v]) => (<React.Fragment key={k}><dt>{k}</dt><dd>{v}</dd></React.Fragment>))}</dl>
      <h2>Experiment results</h2>
      {rows === null && <p className="muted">Loading.</p>}
      {rows?.length === 0 && <p>No results yet. Run: <code>python -m ml.experiments</code></p>}
      {rows?.length > 0 && (
        <div className="scroll">
          <table>
            <thead><tr><th>Suite</th><th>Run</th><th>Accuracy</th><th>Macro-F1</th><th>Train s</th><th>ms / pair</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}><td>{r.suite}</td><td>{r.name}</td><td>{pct(r.accuracy)}</td><td>{r.macro_f1.toFixed(3)}</td>
                  <td>{r.train_seconds.toFixed(1)}</td><td>{r.latency_ms.toFixed(3)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Curve({ history, keys }) {
  const W = 520, H = 220, P = 30;
  const all = history.flatMap((h) => keys.map(([k]) => h[k]));
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(history.length - 1, 1);
  const y = (v) => H - P - ((v - lo) / (hi - lo || 1)) * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training curves">
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--rule)" />
      {keys.map(([k, col, name], j) => (
        <g key={k}>
          <polyline fill="none" stroke={col} strokeWidth="2" points={history.map((h, i) => `${x(i)},${y(h[k])}`).join(" ")} />
          <text x={W - P} y={P + 14 * j} textAnchor="end" fill={col} className="svg-label">{name}</text>
        </g>
      ))}
      <text x={P} y={H - 8} className="svg-label">epoch 1</text>
      <text x={W - P} y={H - 8} textAnchor="end" className="svg-label">epoch {history.length}</text>
    </svg>
  );
}

export function ResearchJournal() {
  const [data, setData] = useState(null);
  const [run, setRun] = useState("");
  useEffect(() => {
    api.reports().then((r) => { setData(r.runs); setRun(r.runs.bilstm_attn ? "bilstm_attn" : Object.keys(r.runs)[0] || ""); }).catch(() => setData({}));
  }, []);
  if (data === null) return <section className="page narrow"><p className="muted">Loading.</p></section>;
  const cur = data[run];
  if (!cur) return <section className="page narrow"><h2>No training runs yet</h2><p>Run <code>python -m ml.train</code> and the curves will appear here.</p></section>;
  const m = cur.metrics, cm = m?.confusion_matrix, labels = m?.labels || [];
  return (
    <section className="page narrow">
      <h2>Training run</h2>
      <select aria-label="Run" value={run} onChange={(e) => setRun(e.target.value)}>
        {Object.keys(data).map((k) => <option key={k}>{k}</option>)}
      </select>
      <Curve history={cur.history} keys={[["train_loss", "var(--refute)", "train loss"], ["dev_loss", "var(--unsure)", "dev loss"], ["dev_macro_f1", "var(--support)", "dev macro-F1"]]} />
      {m && (
        <>
          <p>Test accuracy {pct(m.accuracy)}, macro-F1 {m.macro_f1.toFixed(3)}, precision {m.precision.toFixed(3)}, recall {m.recall.toFixed(3)}.</p>
          <h3>Confusion matrix</h3>
          <p className="hint">Rows are the true label, columns the prediction.</p>
          <table className="cm">
            <thead><tr><th />{labels.map((l) => <th key={l}>{l}</th>)}</tr></thead>
            <tbody>
              {cm.map((row, i) => {
                const tot = row.reduce((a, b) => a + b, 0) || 1;
                return (<tr key={i}><th>{labels[i]}</th>{row.map((v, j) => (
                  <td key={j} style={{ "--a": (v / tot).toFixed(2) }}>{v}</td>))}</tr>);
              })}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
