# -*- coding: utf-8 -*-
"""
web.py — the human front end.

Three things a person needs from a RAG system over their own codebase:

  1. Ask a question and see the answer **with its evidence** (which entities were
     linked, which facts and chunks the model was given, what the verifier said).
  2. Browse the code graph directly — look up a symbol, see its callers, callees
     and exception behaviour, without asking a model anything.
  3. Know whether the index still describes the code on disk — per folder, with
     content hashes and last-built time.

Stdlib only: no Flask, no build step, one HTML page.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import index as idx_mod
from . import llm as llm_mod
from .service import Service


class WebApp:
    def __init__(self, service: Service, chat=None, default_index: str | None = None,
                 top_k: int = 5, max_chars: int = 6000, token: str | None = None):
        self.service = service
        self.chat = chat
        self.default_index = default_index
        self.top_k = top_k
        self.max_chars = max_chars
        self.token = token or None
        self.lock = threading.Lock()

    # ---- helpers ----------------------------------------------------------
    def _index_or_default(self, name):
        return name or self.default_index or (self.service.names() or [None])[0]

    def status(self, deep: bool = True) -> dict:
        rows = self.service.status(deep=deep)
        return {
            "indexes": rows,
            "default_index": self._index_or_default(None),
            "model": {
                "configured": self.chat is not None,
                "base_url": getattr(self.chat, "base_url", None),
                "up": bool(self.chat and self.chat.health()),
                "model": (getattr(self.chat, "model", None) or "").rsplit("/", 1)[-1]
                if self.chat else None,
            },
            "default_top_k": self.top_k,
        }

    def handle(self, path: str, payload: dict | None):
        payload = payload or {}
        if path == "/api/status":
            return self.status(deep=bool(payload.get("deep", True)))
        if path == "/api/search":
            q = str(payload.get("query", "")).strip()
            if not q:
                raise ValueError("empty query")
            return self.service.search(q, self._index_or_default(payload.get("index")),
                                       top_k=int(payload.get("top_k") or 8))
        if path == "/api/entity":
            name = str(payload.get("name", "")).strip()
            if not name:
                raise ValueError("empty entity name")
            return self.service.entity(name, self._index_or_default(payload.get("index")))
        if path == "/api/find":
            pat = str(payload.get("pattern", "")).strip()
            if not pat:
                raise ValueError("empty pattern")
            return self.service.find(pat, self._index_or_default(payload.get("index")))
        if path == "/api/chunk":
            return self.service.chunk(str(payload.get("id", "")),
                                      self._index_or_default(payload.get("index")))
        if path == "/api/answer":
            q = str(payload.get("question", "")).strip()
            if not q:
                raise ValueError("empty question")
            with self.lock:
                return self.service.answer(
                    q, self.chat, self._index_or_default(payload.get("index")),
                    top_k=int(payload.get("top_k") or self.top_k),
                    max_chars=self.max_chars)
        if path == "/api/reindex":
            name = str(payload.get("index", "")).strip()
            if not name:
                raise ValueError("index name required")
            return {"reindexed": self.service.reindex(name)}
        raise LookupError(f"unknown endpoint {path}")


def make_handler(app: WebApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ProdRAG/1.0"

        def log_message(self, fmt, *args):
            sys.stderr.write("  %s\n" % (fmt % args))

        def _send(self, code, body: bytes, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _authorized(self) -> bool:
            if not app.token:
                return True
            got = self.headers.get("X-Prod-RAG-Token") or ""
            if not got:
                from urllib.parse import parse_qs, urlparse
                got = (parse_qs(urlparse(self.path).query).get("token") or [""])[0]
            return got == app.token

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/health":
                self._json(200, {"status": "ok"})
                return
            if not self._authorized():
                self._json(401, {"error": "token required"})
                return
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            payload = {"deep": (q.get("deep", ["1"])[0] not in ("0", "false", "no"))}
            try:
                self._json(200, app.handle(path, payload))
            except LookupError as e:
                self._json(404, {"error": str(e)})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if not self._authorized():
                self._json(401, {"error": "token required"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 200_000:
                    self._json(413, {"error": "request too large"})
                    return
                payload = json.loads(self.rfile.read(n) or b"{}")
                self._json(200, app.handle(path, payload))
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except LookupError as e:
                self._json(404, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PROD_RAG — codebase knowledge</title>
<style>
  :root { --bg:#0d1117; --panel:#151b23; --panel2:#1b222c; --line:#283039;
          --fg:#e6edf3; --dim:#8b949e; --acc:#4c9aff; --ok:#3fb950;
          --warn:#d29922; --bad:#f85149; --mono:ui-monospace,SFMono-Regular,Menlo,monospace; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  header { display:flex; gap:16px; align-items:center; padding:12px 22px;
           border-bottom:1px solid var(--line); background:var(--panel); flex-wrap:wrap; }
  h1 { font-size:15px; margin:0; font-weight:650; letter-spacing:.2px; }
  nav { display:flex; gap:4px; margin-left:auto; }
  nav button { background:transparent; border:1px solid transparent; color:var(--dim);
               padding:6px 14px; border-radius:8px; cursor:pointer; font:inherit; font-size:14px; }
  nav button.on { color:var(--fg); background:var(--panel2); border-color:var(--line); }
  main { max-width:1160px; margin:0 auto; padding:22px; }
  .view { display:none; } .view.on { display:block; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:16px 18px; margin-bottom:16px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.09em;
             color:var(--dim); margin:0 0 12px; font-weight:650; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  input[type=text], textarea, select {
    background:var(--panel2); color:var(--fg); border:1px solid var(--line);
    border-radius:9px; padding:10px 13px; font:inherit; }
  textarea { width:100%; resize:vertical; min-height:64px; }
  input[type=text] { flex:1; min-width:200px; }
  input:focus, textarea:focus, select:focus { outline:2px solid var(--acc); outline-offset:-1px; }
  button.act { background:var(--acc); color:#08131f; border:0; border-radius:9px;
               padding:10px 20px; font-weight:650; font-size:14px; cursor:pointer; }
  button.act:disabled { opacity:.5; cursor:default; }
  button.ghost { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
                 border-radius:8px; padding:6px 13px; font-size:13px; cursor:pointer; }
  .badge { display:inline-block; font-size:12px; font-weight:650; padding:2px 9px;
           border-radius:999px; border:1px solid; }
  .b-supported { color:var(--ok); border-color:var(--ok); }
  .b-corrected { color:var(--acc); border-color:var(--acc); }
  .b-unsupported, .b-correction-rejected { color:var(--bad); border-color:var(--bad); }
  .b-fresh { color:var(--ok); border-color:var(--ok); }
  .b-stale { color:var(--warn); border-color:var(--warn); }
  .b-new, .b-gone, .b-missing, .b-error { color:var(--bad); border-color:var(--bad); }
  .mono { font-family:var(--mono); font-size:12.5px; }
  pre { margin:0; white-space:pre-wrap; word-break:break-word; font-family:var(--mono);
        font-size:12.5px; line-height:1.55; color:#c9d1d9; }
  .answer { font-size:16px; white-space:pre-wrap; }
  .meta { color:var(--dim); font-size:12.5px; margin-top:10px; }
  details { border:1px solid var(--line); border-radius:9px; margin-top:10px; }
  summary { cursor:pointer; padding:9px 13px; color:var(--dim); font-size:13px; }
  summary:hover { color:var(--fg); }
  details > div { padding:0 13px 13px; }
  table { width:100%; border-collapse:collapse; font-size:13.5px; }
  th { text-align:left; color:var(--dim); font-weight:600; font-size:12px;
       text-transform:uppercase; letter-spacing:.06em; padding:6px 8px;
       border-bottom:1px solid var(--line); }
  td { padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
  tr.sub td:first-child { padding-left:26px; }
  .tags { display:flex; gap:7px; flex-wrap:wrap; margin-top:9px; }
  .tag { font-size:12px; background:var(--panel2); border:1px solid var(--line);
         border-radius:7px; padding:2px 8px; cursor:pointer; }
  .tag:hover { border-color:var(--acc); }
  .hint { color:var(--dim); font-size:12.5px; }
  .err { color:var(--bad); white-space:pre-wrap; }
  .split { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:900px){ .split { grid-template-columns:1fr; } }
  .dot { display:inline-block; width:7px; height:7px; border-radius:50%;
         background:var(--bad); margin-right:6px; }
  .dot.up { background:var(--ok); }
  .chunk { border:1px solid var(--line); border-radius:9px; margin-top:9px; overflow:hidden; }
  .chunk > .hd { padding:7px 12px; background:var(--panel2); font-family:var(--mono);
                 font-size:12px; color:var(--dim); }
  .chunk > pre { padding:10px 12px; max-height:340px; overflow:auto; }
  .kv { display:grid; grid-template-columns:130px 1fr; gap:4px 12px; font-size:13.5px; }
  .kv dt { color:var(--dim); } .kv dd { margin:0; }
</style>
</head>
<body>
<header>
  <h1>PROD_RAG</h1>
  <span class="hint" id="hdrinfo"></span>
  <nav>
    <button data-v="ask" class="on">Ask</button>
    <button data-v="browse">Browse</button>
    <button data-v="indexes">Indexes</button>
  </nav>
</header>
<main>
  <!-- ASK -->
  <section class="view on" id="v-ask">
    <div class="card">
      <h2>Question</h2>
      <textarea id="q" placeholder="e.g. What happens if the Server fails to bind to the specified port in run()?"></textarea>
      <div class="row" style="margin-top:10px">
        <button class="act" id="askBtn">Ask</button>
        <select id="askIdx"></select>
        <span class="hint" id="askHint">full path: retrieve → generate → verify</span>
      </div>
    </div>
    <div class="card" id="ansBox" hidden>
      <h2>Answer <span id="ansBadge" class="badge">…</span></h2>
      <div class="answer" id="ans"></div>
      <div class="meta" id="ansMeta"></div>
      <div class="tags" id="ansLinked"></div>
    </div>
    <div class="split" id="ansEv" hidden>
      <div class="card"><h2>Ontology facts the model saw</h2>
        <details open><summary>fact cards</summary><div><pre id="ansFacts"></pre></div></details></div>
      <div class="card"><h2>Retrieved chunks</h2><div id="ansChunks"></div></div>
    </div>
    <div class="card" id="askErr" hidden><h2>Error</h2><div class="err" id="askErrMsg"></div></div>
  </section>

  <!-- BROWSE -->
  <section class="view" id="v-browse">
    <div class="card">
      <h2>Find symbols or search code</h2>
      <div class="row">
        <input type="text" id="bq" placeholder="symbol name, glob (Tcp*), or a natural-language query">
        <select id="bIdx"></select>
        <button class="act" id="findBtn">Find</button>
        <button class="ghost" id="searchBtn">Search code</button>
      </div>
    </div>
    <div class="card" id="bRes" hidden><h2 id="bResTitle">Results</h2><div id="bResBody"></div></div>
    <div class="card" id="entBox" hidden><h2>Entity</h2><div id="entBody"></div></div>
  </section>

  <!-- INDEXES -->
  <section class="view" id="v-idx">
    <div class="card">
      <h2>Indexes</h2>
      <div id="idxBody" class="hint">loading…</div>
    </div>
    <div class="card">
      <h2>Folder freshness</h2>
      <div class="hint" style="margin-bottom:10px">
        Content hashes per folder. A change deep inside a folder marks every ancestor stale.
      </div>
      <div id="folderBody"></div>
    </div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let STATUS = null;
const TOKEN = new URLSearchParams(location.search).get('token') || '';

async function api(path, body) {
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(body)} : {};
  if (TOKEN) opts.headers = Object.assign({}, opts.headers, {'X-Prod-RAG-Token': TOKEN});
  const r = await fetch(path, opts);
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || ('HTTP ' + r.status));
  return d;
}

document.querySelectorAll('nav button').forEach(b => b.onclick = () => {
  document.querySelectorAll('nav button').forEach(x => x.classList.toggle('on', x === b));
  document.querySelectorAll('.view').forEach(v =>
    v.classList.toggle('on', v.id === 'v-' + b.dataset.v));
});

function fillIndexSelects() {
  const names = (STATUS?.indexes || []).map(i => i.name);
  for (const id of ['askIdx','bIdx']) {
    const sel = $(id);
    sel.innerHTML = names.length > 1
      ? '<option value="">all indexes</option>' + names.map(n => `<option>${esc(n)}</option>`).join('')
      : names.map(n => `<option>${esc(n)}</option>`).join('');
    if (STATUS?.default_index) sel.value = STATUS.default_index;
  }
}

async function refresh() {
  STATUS = await api('/api/status');
  const m = STATUS.model;
  $('hdrinfo').innerHTML = (STATUS.indexes || []).length
    ? `${STATUS.indexes.length} index(es) · model: ` +
      (m.configured ? `<span class="dot ${m.up ? 'up' : ''}"></span>${esc(m.model || m.base_url)}`
                    : 'not configured (Q&A disabled)')
    : 'no indexes yet — run <span class="mono">prod-rag index add &lt;name&gt; &lt;path&gt;</span>';
  fillIndexSelects();
  renderIndexes();
}

function fmtStats(s) { return s ? `${s.files} files · ${s.entities} entities · ${s.chunks} chunks` : ''; }

function renderIndexes() {
  const rows = STATUS.indexes || [];
  if (!rows.length) { $('idxBody').textContent = 'No indexes built.'; $('folderBody').textContent = ''; return; }
  let h = '<table><tr><th>name</th><th>root</th><th>size</th><th>built</th><th>state</th><th></th></tr>';
  for (const r of rows) {
    const f = r.freshness || {};
    h += `<tr><td><b>${esc(r.name)}</b><div class="hint">${esc(r.label)}</div></td>
      <td class="mono">${esc(r.root)}</td><td>${fmtStats(r.stats)}</td>
      <td class="mono">${esc(r.built_at)}</td>
      <td><span class="badge b-${esc(f.state||'')}">${esc(f.state||'?')}</span>
        <div class="hint">${esc(f.detail||'')}</div></td>
      <td><button class="ghost" onclick="reindex('${esc(r.name)}')">Rebuild</button></td></tr>`;
  }
  $('idxBody').innerHTML = h + '</table>';

  let g = '';
  for (const r of rows) {
    const f = r.freshness || {};
    const folders = f.folders || [];
    g += `<h2 style="margin-top:14px">${esc(r.name)} <span class="hint">${esc(r.root)}</span></h2>`;
    g += '<table><tr><th>folder</th><th>files</th><th>subtree</th><th>hash</th><th>status</th></tr>';
    for (const d of folders) {
      const cls = d.status === 'fresh' ? '' : 'sub';
      const hash = d.hash === d.hash_now ? d.hash : `${d.hash} → <span style="color:var(--warn)">${d.hash_now}</span>`;
      g += `<tr class="${cls}"><td class="mono">${esc(d.path)}</td><td>${d.files}</td>
        <td>${d.subtree_files}</td><td class="mono">${hash}</td>
        <td><span class="badge b-${esc(d.status)}">${esc(d.status)}</span></td></tr>`;
    }
    g += '</table>';
    if ((f.changed||[]).length || (f.added||[]).length || (f.removed||[]).length) {
      g += '<details><summary>changed files</summary><div><pre>';
      if ((f.changed||[]).length) g += 'modified:\n  ' + f.changed.map(esc).join('\n  ') + '\n';
      if ((f.added||[]).length) g += 'added:\n  ' + f.added.map(esc).join('\n  ') + '\n';
      if ((f.removed||[]).length) g += 'removed:\n  ' + f.removed.map(esc).join('\n  ');
      g += '</pre></div></details>';
    }
  }
  $('folderBody').innerHTML = g;
}

async function reindex(name) {
  try {
    const r = await api('/api/reindex', {index: name});
    await refresh();
    alert('Rebuilt ' + name);
  } catch (e) { alert('Rebuild failed: ' + e.message); }
}

/* ---------- ask ---------- */
$('askBtn').onclick = async () => {
  const q = $('q').value.trim();
  if (!q) return;
  const btn = $('askBtn'); btn.disabled = true; btn.textContent = 'Thinking…';
  $('askErr').hidden = true; $('ansBox').hidden = false; $('ansEv').hidden = false;
  $('ans').textContent = ''; $('ansFacts').textContent = ''; $('ansChunks').innerHTML = '';
  $('ansBadge').className = 'badge'; $('ansBadge').textContent = 'verifying…';
  $('ansMeta').textContent = ''; $('ansLinked').innerHTML = '';
  const t0 = performance.now();
  try {
    const d = await api('/api/answer', {question: q, index: $('askIdx').value, top_k: 5});
    const wall = ((performance.now() - t0) / 1000).toFixed(1);
    $('ans').textContent = d.answer;
    const flag = d.verification.flag || 'unsupported';
    $('ansBadge').className = 'badge b-' + flag; $('ansBadge').textContent = flag;
    const t = d.timings || {}, m = t.model || {};
    const bits = [`${wall}s`, `index ${d.index}`, d.model || ''];
    if (m.prompt_per_second) bits.push(`${m.prompt_per_second.toFixed(0)} tok/s prefill`);
    if (m.predicted_per_second) bits.push(`${m.predicted_per_second.toFixed(0)} tok/s decode`);
    $('ansMeta').textContent = bits.filter(Boolean).join(' · ');
    const claims = d.verification.claims || [];
    if (claims.length) $('ansMeta').textContent += ' · verifier: ' + claims.join(' | ');
    for (const e of (d.linked || []))
      $('ansLinked').insertAdjacentHTML('beforeend',
        `<span class="tag" onclick="openEntity('${esc(e.qname)}')">${esc(e.qname)}</span>`);
    $('ansFacts').textContent = d.facts || '(no ontology facts matched)';
    $('ansChunks').innerHTML = (d.chunks || []).map(c =>
      `<div class="chunk"><div class="hd">${esc(c.kind)} ${esc(c.qname)} · ${esc(c.file)}:${c.line_start}-${c.line_end}</div>
       <pre>${esc(c.text || '')}</pre></div>`).join('') || '<div class="hint">none</div>';
  } catch (e) {
    $('askErr').hidden = false; $('askErrMsg').textContent = e.message;
    $('ansBadge').className = 'badge b-unsupported'; $('ansBadge').textContent = 'error';
  } finally { btn.disabled = false; btn.textContent = 'Ask'; }
};
$('q').addEventListener('keydown', e => { if ((e.ctrlKey||e.metaKey) && e.key === 'Enter') $('askBtn').click(); });

/* ---------- browse ---------- */
$('findBtn').onclick = async () => {
  const p = $('bq').value.trim(); if (!p) return;
  try {
    const d = await api('/api/find', {pattern: p, index: $('bIdx').value});
    $('bRes').hidden = false; $('bResTitle').textContent = `Symbols matching "${p}" (${d.count})`;
    $('bResBody').innerHTML = (d.matches || []).map(m =>
      `<div class="row" style="padding:5px 0;border-bottom:1px solid var(--line)">
         <span class="tag" onclick="openEntity('${esc(m.qname)}')">${esc(m.qname)}</span>
         <span class="hint">${esc(m.kind)} · ${esc(m.file)}:${m.line_start}</span></div>`).join('')
      || '<div class="hint">no matches</div>';
  } catch (e) { alert(e.message); }
};
$('searchBtn').onclick = async () => {
  const q = $('bq').value.trim(); if (!q) return;
  try {
    const d = await api('/api/search', {query: q, index: $('bIdx').value, top_k: 8});
    const groups = d.indexes || [d];
    $('bRes').hidden = false; $('bResTitle').textContent = `Code search: "${q}"`;
    $('bResBody').innerHTML = groups.map(g => `
      <h3 class="hint" style="margin:10px 0 4px">${esc(g.index)} — ${esc(g.root||'')}</h3>
      <details open><summary>ontology facts</summary><div><pre>${esc(g.facts || '(none)')}</pre></div></details>
      ${(g.chunks||[]).map(c => `<div class="chunk">
         <div class="hd">${esc(c.kind)} ${esc(c.qname)} · ${esc(c.file)}:${c.line_start}-${c.line_end}</div>
         <pre>${esc(c.text || '')}</pre></div>`).join('')}`).join('');
  } catch (e) { alert(e.message); }
};

async function openEntity(name) {
  document.querySelector('nav button[data-v=browse]').click();
  try {
    const d = await api('/api/entity', {name, index: $('bIdx').value});
    $('entBox').hidden = false;
    if (!d.found) { $('entBody').innerHTML = `<div class="hint">not found: ${esc(name)}</div>`; return; }
    if (d.ambiguous) {
      $('entBody').innerHTML = '<div class="hint">ambiguous — pick one:</div>' +
        d.matches.map(m => `<div><span class="tag" onclick="openEntity('${esc(m.qname)}')">${esc(m.qname)}</span>
          <span class="hint">${esc(m.index)} · ${esc(m.file)}</span></div>`).join('');
      return;
    }
    const list = (a) => (a && a.length) ? a.map(esc).join(', ') : '—';
    $('entBody').innerHTML = `
      <div class="row" style="margin-bottom:10px">
        <b class="mono">${esc(d.qname)}</b>
        <span class="hint">${esc(d.kind)} · ${esc(d.file)}:${d.line_start}-${d.line_end}</span>
      </div>
      <dl class="kv">
        <dt>signature</dt><dd class="mono">${esc(d.fact_card)}</dd>
        <dt>returns</dt><dd>${esc(d.return_type || '—')}</dd>
        <dt>modifiers</dt><dd>${esc((d.modifiers||[]).join(', ') || '—')}</dd>
        <dt>try/catch</dt><dd>${d.has_try_catch ? '<span class="badge b-supported">yes</span>' : '<span class="badge b-stale">no</span>'}</dd>
        <dt>throws</dt><dd>${(d.throw_sites||[]).map(s => `${esc(s.type)} @${s.line}${s.guarded ? ' (guarded)' : ' (unguarded)'}`).join('<br>') || '—'}</dd>
        <dt>calls</dt><dd class="mono">${list(d.calls)}</dd>
        <dt>called by</dt><dd class="mono">${list(d.callers)}</dd>
        <dt>inherits</dt><dd class="mono">${list(d.inherits)}</dd>
        <dt>inherited by</dt><dd class="mono">${list(d.inherited_by)}</dd>
        <dt>members</dt><dd class="mono">${list(d.members)}</dd>
      </dl>
      <details style="margin-top:12px"><summary>source</summary><div><pre>${esc(d.code || '')}</pre></div></details>`;
  } catch (e) { alert(e.message); }
}

refresh();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prod-rag-web", description="PROD_RAG web UI")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 0.0.0.0 to share on the LAN")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--index", action="append", metavar="NAME",
                    help="load only these indexes (default: all)")
    ap.add_argument("--llm-base-url", default=os.environ.get("PROD_RAG_LLM_BASE_URL"),
                    help="OpenAI-compatible endpoint for the Q&A box")
    ap.add_argument("--llm-model", default=os.environ.get("PROD_RAG_LLM_MODEL"))
    ap.add_argument("--llm-api-key", default=os.environ.get("PROD_RAG_LLM_API_KEY"))
    ap.add_argument("--token", default=os.environ.get("PROD_RAG_TOKEN"),
                    help="shared secret required by the API (recommended with --host 0.0.0.0)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args(argv)

    service = Service(args.index)
    chat = llm_mod.client_from_env(args.llm_base_url, args.llm_model, args.llm_api_key)
    if chat is not None:
        chat.resolve_model()

    app = WebApp(service, chat, default_index=(args.index[0] if args.index else None),
                 top_k=args.top_k, max_chars=args.max_chars, token=args.token)

    print(f"indexes: {', '.join(service.names()) or '(none)'}")
    print(f"model:   {chat.base_url if chat else 'not configured (Q&A disabled)'}")
    if args.host in ("0.0.0.0", "::") and not args.token:
        print("warning: bound to all interfaces without --token; anyone on the "
              "network can query the index", file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    suffix = f"?token={args.token}" if args.token else ""
    print(f"\n  http://{shown}:{args.port}/{suffix}   (Ctrl+C to stop)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
