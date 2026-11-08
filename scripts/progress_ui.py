"""Tiny live progress dashboard for a running benchmark.

    python scripts/progress_ui.py --backend neural --port 8765
    -> open http://localhost:8765

Reads results/<backend>/run_seed0.log and the stage cache files written by
src/eval/staged.py; no dependencies beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>TriSafe-RAG run</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
h1{font-size:18px;margin:0 0 4px}.sub{color:#94a3b8;font-size:13px;margin-bottom:20px}
.card{background:#1e293b;border-radius:10px;padding:14px 16px;margin-bottom:12px}
.row{display:flex;justify-content:space-between;font-size:14px;margin-bottom:6px}
.bar{height:10px;background:#334155;border-radius:5px;overflow:hidden}
.fill{height:100%;background:#38bdf8;transition:width .6s}
.done .fill{background:#4ade80}.pending{opacity:.55}
.kv{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;font-size:13px}
.kv div span{display:block;color:#94a3b8;font-size:11px}
.foot{color:#64748b;font-size:12px;margin-top:14px}
pre{background:#0b1220;padding:10px;border-radius:8px;font-size:11px;color:#a5b4fc;white-space:pre-wrap}
</style></head><body>
<h1>TriSafe-RAG benchmark run</h1><div class="sub" id="sub"></div>
<div id="stages"></div>
<div class="card kv" id="kv"></div>
<div class="card"><pre id="log"></pre></div>
<div class="foot" id="foot"></div>
<script>
const names={translate:"1. Translate (LLM)",retrieve:"2. Retrieve (e5 + BM25)",generate:"3. Generate drafts (LLM)",ground:"4. Ground (NLI)",final:"5. Replay, bootstrap, results.json"};
async function tick(){
  const r=await fetch('/progress.json',{cache:'no-store'}); const p=await r.json();
  document.getElementById('sub').textContent=p.backend+' backend, seed '+p.seed+(p.alive?' - process running':' - process not running')+' - '+p.now;
  let h='';
  for(const s of p.stages){const pct=s.total?Math.round(100*s.done/s.total):0;
    h+=`<div class="card ${s.done>=s.total&&s.total?'done':''} ${s.done===0?'pending':''}"><div class="row"><b>${names[s.name]}</b><span>${s.done} / ${s.total} (${pct}%)</span></div><div class="bar"><div class="fill" style="width:${pct}%"></div></div></div>`}
  document.getElementById('stages').innerHTML=h;
  const kv=[['Current stage',p.current],['Rate',p.rate?p.rate.toFixed(1)+' s / item':'-'],['ETA (this stage)',p.eta_min!=null?p.eta_min.toFixed(0)+' min':'-'],['Est. finish (all)',p.eta_total||'-'],['Memory free',p.mem_free_gb!=null?p.mem_free_gb.toFixed(1)+' GB':'-'],['Elapsed',p.elapsed],['Results file',p.results?'written':'pending'],['Last log line',p.last_time]];
  document.getElementById('kv').innerHTML=kv.map(([k,v])=>`<div><span>${k}</span>${v}</div>`).join('');
  document.getElementById('log').textContent=p.tail.join('\\n');
  document.getElementById('foot').textContent='Refreshes every 5 s. '+(p.results?'Done: run scripts/make_tables.py and scripts/make_overleaf_zip.py.':'');
}
tick(); setInterval(tick,5000);
</script></body></html>"""


def read_json_len(p: Path) -> int:
    try:
        return len(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return 0


def progress(backend: str, seed: int, n_queries: int, n_trans: int) -> dict:
    out_dir = ROOT / "results" / backend
    sdir = out_dir / f"stages_seed{seed}"
    log_path = out_dir / f"run_seed{seed}.log"
    lines = []
    if log_path.exists():
        raw = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n").splitlines()
        lines = [l for l in raw if l.strip() and "Warning" not in l and "Batches" not in l and "Sliding" not in l]
    counts = {
        "translate": (read_json_len(sdir / "translations.json"), n_trans),
        "retrieve": (read_json_len(sdir / "retrieval.json") + read_json_len(sdir / "retrieval_translated.json"), n_queries + n_trans),
        "generate": (read_json_len(sdir / "generation.json") + read_json_len(sdir / "generation_translated.json"), n_queries + n_trans),
        "ground": (read_json_len(sdir / "grounding.json") + read_json_len(sdir / "grounding_translated.json"), n_queries + n_trans),
    }
    results = (out_dir / "results.json").exists()
    counts["final"] = (1 if results else 0, 1)
    current = "finished" if results else next((k for k, (d, t) in counts.items() if d < t), "final")
    rate = eta = None
    for l in reversed(lines):
        m = re.search(r"\((\d+\.\d)s each, ~(\d+) min left\)", l)
        if m and current in l:
            rate, eta = float(m.group(1)), float(m.group(2))
            break
    # rough total ETA: remaining items of current stage + fixed allowances for later stages
    allowances = {"translate": 0, "retrieve": 1, "generate": 0, "ground": 18, "final": 10}
    total_min = None
    if not results:
        order = list(counts)
        idx = order.index(current)
        total_min = (eta or 0) + sum(allowances[s] for s in order[idx + 1:])
    try:
        import psutil
        mem = psutil.virtual_memory().available / 1e9
        alive = any("run_benchmark.py" in " ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))
    except Exception:
        mem, alive = None, None
    start = None
    for l in lines:
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", l)
        if m:
            start = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            break
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start)) if start else "-"
    last_time = next((re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", l).group(1)[11:] for l in reversed(lines) if re.match(r"\d{4}-", l)), "-")
    return {
        "backend": backend, "seed": seed, "now": time.strftime("%H:%M:%S"), "alive": alive,
        "stages": [{"name": k, "done": d, "total": t} for k, (d, t) in counts.items()],
        "current": current, "rate": rate, "eta_min": eta,
        "eta_total": time.strftime("%H:%M", time.localtime(time.time() + 60 * total_min)) if total_min is not None else None,
        "mem_free_gb": mem, "elapsed": elapsed, "results": results, "last_time": last_time,
        "tail": lines[-6:],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="neural")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    queries = json.loads((ROOT / "data" / "benchmark" / "queries.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "data" / "benchmark" / "split.json").read_text(encoding="utf-8"))
    test = set(split["test"])
    n_q = len(queries)
    n_t = sum(1 for q in queries if q["language"] != "en" and q["id"] in test)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/progress.json"):
                body = json.dumps(progress(args.backend, args.seed, n_q, n_t)).encode("utf-8")
                ctype = "application/json"
            else:
                body = HTML.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    print(f"progress UI at http://localhost:{args.port}  (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
