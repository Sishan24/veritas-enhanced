async function call(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

export const health = () => call("/api/health");
export const reports = () => call("/api/reports");
export const analyze = (text, evidence) =>
  call("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, evidence: evidence.length ? evidence : null }),
  });
