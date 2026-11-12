import React, { useEffect, useState } from "react";
import * as api from "./api.js";

/* ─── Constants ────────────────────────────────────────────────── */

export const VERDICT = {
  SUPPORTED:    { label: "Supported",              color: "var(--support)",  icon: "✓" },
  REFUTED:      { label: "Refuted",                color: "var(--refute)",   icon: "✗" },
  CONFLICTING:  { label: "Conflicting evidence",   color: "var(--conflict)", icon: "⇆" },
  INSUFFICIENT: { label: "Insufficient evidence",  color: "var(--unsure)",   icon: "?" },
};

const STANCE = {
  SUPPORTED:      "var(--support)",
  REFUTED:        "var(--refute)",
  NOT_ENOUGH_INFO:"var(--unsure)",
};

const STANCE_LABEL = {
  SUPPORTED:      "Supports",
  REFUTED:        "Contradicts",
  NOT_ENOUGH_INFO:"Not enough to say",
};

const pct = (x) => `${Math.round(x * 100)}%`;

const SAMPLE = `The United Nations Security Council held an emergency session on the Gaza ceasefire proposal. Spain won the UEFA Euro 2024 championship. The US Federal Reserve raised interest rates to a 22-year high. Global CO2 emissions from fossil fuels reached a new record in 2023. No credible evidence supports claims that 5G towers spread COVID-19. Bitcoin surpassed 70 000 dollars in price in 2024. Canada experienced its worst wildfire season on record in 2023.`;

/* ─── Shared Components ────────────────────────────────────────── */

function Empty({ go }) {
  return (
    <section className="page narrow">
      <div className="empty-state">
        <div className="empty-icon">🔍</div>
        <h2>Nothing analysed yet</h2>
        <p>
          Paste an article or statement in Claim Studio and VERITAS will check
          each claim against the evidence corpus.
        </p>
        <button className="primary" onClick={() => go("studio")}>
          Open Claim Studio →
        </button>
      </div>
    </section>
  );
}

function VerdictTag({ v }) {
  const d = VERDICT[v];
  return <span className="tag" style={{ "--c": d.color }}>{d.label}</span>;
}

function ClaimPicker({ result, selected, setSelected }) {
  return (
    <div>
      <h4>Claims ({result.claims.length})</h4>
      <ol className="claims">
        {result.claims.map((c, i) => (
          <li key={c.id}>
            <button
              className={i === selected ? "claim on" : "claim"}
              onClick={() => setSelected(i)}
            >
              <span>{c.text}</span>
              <VerdictTag v={c.verdict} />
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}

/* ─── 1. Archive (Landing) ─────────────────────────────────────── */

export function Archive({ status, go }) {
  return (
    <section className="page hero">
      <div className="hero-badge">
        <span className="dot" />
        Evidence-Aware Deep Learning Framework
      </div>

      <h1>What does this claim actually rest on?</h1>

      <p className="lede">
        Paste an article or statement. VERITAS splits it into individual claims,
        retrieves the passages that back or contradict each one, and shows which
        words in the evidence drove the verdict — with word-level attention
        weights.
      </p>

      <div className="verdicts">
        {Object.entries(VERDICT).map(([k, v]) => (
          <span key={k} className="tag" style={{ "--c": v.color }}>
            {v.label}
          </span>
        ))}
      </div>

      <div className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon">🔬</div>
          <h4>Claim Extraction</h4>
          <p>Rule-based sentence segmentation with claim-worthiness heuristics</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">📑</div>
          <h4>Evidence Retrieval</h4>
          <p>TF-IDF similarity search over evidence passages with n-gram matching</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">🧠</div>
          <h4>Stance Detection</h4>
          <p>BiLSTM + Attention classifies each claim–evidence pair into three stances</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">💡</div>
          <h4>Explainability</h4>
          <p>Word-level attention weights show which tokens drove the decision</p>
        </div>
      </div>

      <button className="primary" onClick={() => go("studio")}>
        Open Claim Studio →
      </button>

      <div className="status" style={{ marginTop: 24 }}>
        {status === null && "⏳ Checking the analysis server…"}
        {status?.offline && "⚠ The analysis server is not running. Start it with: python -m uvicorn backend.app.main:app --reload"}
        {status && !status.offline && !status.model_ready && "⚠ Server is running, but no model is trained yet. Run: python -m ml.make_sample_data, then python -m ml.train"}
        {status?.model_ready && `✓ Ready — ${status.arch} model loaded, ${status.default_evidence} evidence passages indexed.`}
      </div>
    </section>
  );
}

/* ─── 2. Claim Studio ──────────────────────────────────────────── */

export function ClaimStudio({ run, busy, error, status }) {
  const [text, setText] = useState(SAMPLE);
  const [evidence, setEvidence] = useState([]);

  function onFiles(e) {
    Promise.all(
      [...e.target.files].map((f) =>
        f.text().then((t) => ({ id: f.name, text: t }))
      )
    ).then((docs) => setEvidence((prev) => [...prev, ...docs]));
  }

  return (
    <section className="page studio">
      <div className="studio-header">
        <h2>Claim Studio</h2>
      </div>

      <div className="corpus-notice">
        <span className="corpus-notice-icon">ℹ</span>
        <span>Claims are verified against the loaded evidence corpus. This system does not search the web — it works entirely with local evidence.</span>
      </div>

      <div>
        <label htmlFor="claims">Article or statement</label>
        <textarea
          id="claims"
          rows={10}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste any text containing factual claims..."
        />
      </div>

      <div>
        <label>Evidence documents (optional)</label>
        <div className="file-upload">
          <input
            id="ev"
            type="file"
            accept=".txt,.md"
            multiple
            onChange={onFiles}
          />
          <div className="file-upload-icon">📂</div>
          <p>
            {evidence.length
              ? `${evidence.length} file(s) added`
              : "Drop .txt or .md files here, or click to browse"}
          </p>
        </div>

        {evidence.length > 0 && (
          <div className="file-chips">
            {evidence.map((d) => (
              <span key={d.id} className="file-chip">📄 {d.id}</span>
            ))}
            <button className="quiet" onClick={() => setEvidence([])}>
              Remove all
            </button>
          </div>
        )}

        {evidence.length === 0 && (
          <p className="hint" style={{ marginTop: 8 }}>
            With no files, claims are checked against the built-in sample evidence corpus.
          </p>
        )}
      </div>

      {error && <p className="error" role="alert">⚠ {error}</p>}

      <button
        className={`primary ${busy ? "loading" : ""}`}
        disabled={busy || !text.trim() || status?.model_ready === false}
        onClick={() => run(text, evidence)}
      >
        {busy ? "Analysing…" : "Analyse claims →"}
      </button>
    </section>
  );
}

/* ─── 3. Evidence Chamber ──────────────────────────────────────── */

function Highlighted({ tokens }) {
  if (!tokens)
    return (
      <span className="muted">
        This model does not expose attention weights. Use bilstm_attn for
        word-level evidence.
      </span>
    );
  const max = Math.max(...tokens.map((t) => t.weight), 1e-9);
  return (
    <p className="attn">
      {tokens.map((t, i) => (
        <mark
          key={i}
          style={{ "--a": (t.weight / max).toFixed(2) }}
          title={`attention weight: ${t.weight.toFixed(4)}`}
        >
          {t.token}{" "}
        </mark>
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
          <VerdictTag v={claim.verdict} />
          <span>
            {pct(claim.confidence)} confidence · {claim.evidence.length} retrieved passage(s)
          </span>
        </p>
        {claim.evidence.length === 0 && (
          <p className="muted" style={{ marginTop: 16 }}>
            No evidence passages were found for this claim in the corpus.
          </p>
        )}
        {claim.evidence.map((e) => (
          <article
            key={e.id}
            className="card"
            style={{ "--c": STANCE[e.stance] }}
          >
            <header>
              <strong>{STANCE_LABEL[e.stance]}</strong>
              <span className="muted">
                {pct(e.probs[e.stance])} · match {e.retrieval_score.toFixed(2)} · {e.id}
              </span>
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

/* ─── 4. Context Graph ─────────────────────────────────────────── */

export function ContextGraph({ result, selected, setSelected, go }) {
  if (!result?.claims.length) return <Empty go={go} />;
  const claim = result.claims[selected];
  const n = claim.evidence.length;
  const R = 150, cx = 260, cy = 210;
  return (
    <section className="page chamber">
      <ClaimPicker {...{ result, selected, setSelected }} />
      <div>
        <h2>Evidence Constellation</h2>
        <figure>
          <svg
            viewBox="0 0 520 420"
            role="img"
            aria-label={`Evidence constellation for: ${claim.text}`}
          >
            {/* Background grid */}
            <circle cx={cx} cy={cy} r={R + 40} fill="none" stroke="var(--border-subtle)" strokeDasharray="4 4" />
            <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--border-subtle)" strokeDasharray="4 4" />
            <circle cx={cx} cy={cy} r={R - 50} fill="none" stroke="var(--border-subtle)" strokeDasharray="4 4" />

            {claim.evidence.map((e, i) => {
              const a = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2;
              const x = cx + R * Math.cos(a),
                    y = cy + R * Math.sin(a);
              const col = STANCE[e.stance];
              return (
                <g key={e.id}>
                  <line
                    x1={cx} y1={cy} x2={x} y2={y}
                    stroke={col}
                    strokeWidth={1 + 5 * e.probs[e.stance]}
                    strokeDasharray={e.stance === "NOT_ENOUGH_INFO" ? "5 5" : undefined}
                    opacity="0.7"
                  />
                  <circle cx={x} cy={y} r={8 + 12 * e.retrieval_score} fill={col} opacity="0.85" />
                  <circle cx={x} cy={y} r={8 + 12 * e.retrieval_score + 4} fill="none" stroke={col} strokeWidth="1" opacity="0.3" />
                  <text
                    x={x}
                    y={y + (y < cy ? -26 : 34)}
                    textAnchor="middle"
                    className="svg-label"
                  >
                    {e.id}
                  </text>
                </g>
              );
            })}

            {/* Central claim node */}
            <circle cx={cx} cy={cy} r="34" fill={VERDICT[claim.verdict].color} opacity="0.9" />
            <circle cx={cx} cy={cy} r="38" fill="none" stroke={VERDICT[claim.verdict].color} strokeWidth="1" opacity="0.3" />
            <text x={cx} y={cy + 4} textAnchor="middle" className="svg-claim">
              claim
            </text>
          </svg>
          <figcaption>
            Line colour → stance. Line weight → model certainty. Node size → passage–claim similarity.
            Dashed lines → insufficient evidence.
          </figcaption>
        </figure>
      </div>
    </section>
  );
}

/* ─── 5. Stance Observatory ────────────────────────────────────── */

export function StanceObservatory({ result, go }) {
  if (!result?.claims.length) return <Empty go={go} />;
  const counts = result.claims.reduce(
    (a, c) => ({ ...a, [c.verdict]: (a[c.verdict] || 0) + 1 }),
    {}
  );
  return (
    <section className="page narrow">
      <h2>{result.claims.length} claims analysed</h2>
      <p className="muted" style={{ marginBottom: 16 }}>Verdict distribution across all claims</p>

      <div className="stack" role="img" aria-label="Verdict distribution">
        {Object.keys(VERDICT)
          .filter((k) => counts[k])
          .map((k) => (
            <div
              key={k}
              style={{ flex: counts[k], background: VERDICT[k].color }}
            >
              {counts[k]} {VERDICT[k].label.toLowerCase()}
            </div>
          ))}
      </div>

      {result.claims.map((c) => (
        <div key={c.id} className="obs">
          <p>
            {c.text} <VerdictTag v={c.verdict} />
          </p>
          {c.evidence.map((e) => (
            <div key={e.id} className="bars" title={e.text}>
              <span>{e.id}</span>
              <div className="stack thin">
                {Object.keys(STANCE).map((k) => (
                  <div
                    key={k}
                    style={{ flex: e.probs[k], background: STANCE[k] }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}
      <p className="hint" style={{ marginTop: 20 }}>
        Bars show each passage's probability split: supports (green),
        contradicts (red), not enough to say (ochre).
      </p>
    </section>
  );
}

/* ─── 6. Neural Lab ────────────────────────────────────────────── */

const ARCH_NOTES = [
  ["mlp",         "MLP",              "Averages word vectors, then classifies. The baseline that ignores word order."],
  ["cnn",         "CNN",              "Convolutions over 2–4 word windows capture local phrasing patterns."],
  ["lstm",        "LSTM",             "Reads the text left-to-right, carrying context forward through gates."],
  ["bilstm",      "BiLSTM",           "Reads in both directions so each word sees what comes before and after."],
  ["bilstm_attn", "BiLSTM + Attn",    "BiLSTM plus claim-guided attention: weights evidence words by relevance. Powers the attention highlights."],
];

export function NeuralLab() {
  const [rows, setRows] = useState(null);
  useEffect(() => {
    api.reports().then((r) => setRows(r.comparison)).catch(() => setRows([]));
  }, []);

  return (
    <section className="page narrow">
      <h2>Architecture Comparison</h2>
      <p className="muted" style={{ marginBottom: 20 }}>
        Five neural architectures compared on the same claim–evidence data.
      </p>

      <dl className="arch">
        {ARCH_NOTES.map(([k, name, desc]) => (
          <React.Fragment key={k}>
            <dt>{name}</dt>
            <dd>{desc}</dd>
          </React.Fragment>
        ))}
      </dl>

      <h2>Experiment Results</h2>
      {rows === null && (
        <div className="skeleton" style={{ height: 200, width: "100%" }} />
      )}
      {rows?.length === 0 && (
        <p className="muted">
          No results yet. Run: <code>python -m ml.experiments</code>
        </p>
      )}
      {rows?.length > 0 && (
        <div className="scroll">
          <table>
            <thead>
              <tr>
                <th>Suite</th>
                <th>Run</th>
                <th>Accuracy</th>
                <th>Macro-F1</th>
                <th>Train (s)</th>
                <th>Latency (ms)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.suite}</td>
                  <td style={{ fontFamily: "var(--font-mono)", fontSize: 12 }}>{r.name}</td>
                  <td>{pct(r.accuracy)}</td>
                  <td>{r.macro_f1.toFixed(3)}</td>
                  <td>{r.train_seconds.toFixed(1)}</td>
                  <td>{r.latency_ms.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

/* ─── 7. Research Journal ──────────────────────────────────────── */

function Curve({ history, keys }) {
  const W = 560, H = 240, P = 40;
  const all = history.flatMap((h) => keys.map(([k]) => h[k]));
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(history.length - 1, 1);
  const y = (v) => H - P - ((v - lo) / (hi - lo || 1)) * (H - 2 * P);

  return (
    <div className="curve-container">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training curves">
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={P} y1={P + f * (H - 2 * P)} x2={W - P} y2={P + f * (H - 2 * P)}
            stroke="var(--border-subtle)" strokeDasharray="4 4" />
        ))}
        <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--border-medium)" />
        <line x1={P} y1={P} x2={P} y2={H - P} stroke="var(--border-medium)" />

        {keys.map(([k, col, name], j) => (
          <g key={k}>
            <polyline
              fill="none"
              stroke={col}
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={history.map((h, i) => `${x(i)},${y(h[k])}`).join(" ")}
            />
            {/* Data points */}
            {history.map((h, i) => (
              <circle key={i} cx={x(i)} cy={y(h[k])} r="3" fill={col} opacity="0.7" />
            ))}
            <text
              x={W - P + 8}
              y={y(history[history.length - 1][k]) + 4}
              fill={col}
              className="svg-label"
              textAnchor="start"
            >
              {name}
            </text>
          </g>
        ))}
        <text x={P} y={H - 10} className="svg-label">epoch 1</text>
        <text x={W - P} y={H - 10} textAnchor="end" className="svg-label">
          epoch {history.length}
        </text>
      </svg>
    </div>
  );
}

export function ResearchJournal() {
  const [data, setData] = useState(null);
  const [run, setRun] = useState("");
  useEffect(() => {
    api
      .reports()
      .then((r) => {
        setData(r.runs);
        setRun(
          r.runs.bilstm_attn
            ? "bilstm_attn"
            : Object.keys(r.runs)[0] || ""
        );
      })
      .catch(() => setData({}));
  }, []);

  if (data === null)
    return (
      <section className="page narrow">
        <div className="skeleton" style={{ height: 300, width: "100%" }} />
      </section>
    );

  const cur = data[run];
  if (!cur)
    return (
      <section className="page narrow">
        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h2>No training runs yet</h2>
          <p>
            Run <code>python -m ml.train</code> and the training curves and
            metrics will appear here.
          </p>
        </div>
      </section>
    );

  const m = cur.metrics,
    cm = m?.confusion_matrix,
    labels = m?.labels || [];

  return (
    <section className="page narrow">
      <h2>Training Run</h2>
      <select
        aria-label="Run"
        value={run}
        onChange={(e) => setRun(e.target.value)}
      >
        {Object.keys(data).map((k) => (
          <option key={k}>{k}</option>
        ))}
      </select>

      <Curve
        history={cur.history}
        keys={[
          ["train_loss", "var(--refute)", "train loss"],
          ["dev_loss", "var(--unsure)", "dev loss"],
          ["dev_macro_f1", "var(--support)", "dev F1"],
        ]}
      />

      {m && (
        <>
          <div className="metric-grid">
            <div className="metric-card">
              <div className="metric-value">{pct(m.accuracy)}</div>
              <div className="metric-label">Accuracy</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{m.macro_f1.toFixed(3)}</div>
              <div className="metric-label">Macro-F1</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{m.precision.toFixed(3)}</div>
              <div className="metric-label">Precision</div>
            </div>
            <div className="metric-card">
              <div className="metric-value">{m.recall.toFixed(3)}</div>
              <div className="metric-label">Recall</div>
            </div>
          </div>

          <h3>Confusion Matrix</h3>
          <p className="hint">Rows = true label · Columns = prediction</p>
          <table className="cm">
            <thead>
              <tr>
                <th />
                {labels.map((l) => (
                  <th key={l}>{l.slice(0, 3)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.map((row, i) => {
                const tot = row.reduce((a, b) => a + b, 0) || 1;
                return (
                  <tr key={i}>
                    <th>{labels[i]}</th>
                    {row.map((v, j) => (
                      <td
                        key={j}
                        style={{ "--a": (v / tot).toFixed(2) }}
                      >
                        {v}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
