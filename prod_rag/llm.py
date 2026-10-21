# -*- coding: utf-8 -*-
"""
llm.py — optional model backend for the Q&A box.

Everything else in PROD_RAG works without a model: the MCP tools return
structured evidence and the agent composes the answer. This module exists for
the one case that needs a model of our own — the web UI's Q&A box, which runs
the full retrieve -> generate -> verify -> (correct) loop.

Any OpenAI-compatible endpoint works (llama.cpp, vLLM, Ollama, LM Studio,
DeepSeek). No key is required for the local ones.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from . import core as oc

SYSTEM_PROMPT = """You are a precise assistant answering questions about C++ code.
You are given:
  1. ONTOLOGY FACTS - exact structured facts (signatures, parameters, calls,
     fields, exceptions, and try/catch presence). Trust them literally.
  2. RELEVANT CODE CHUNKS - real source code.

Hard rules:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. NEVER state a value, number, default, or flag unless it appears LITERALLY in
   the context. If the context does not give it, say it is not specified.
3. For error/exception questions use the ONTOLOGY FACTS. If a fact says
   "NO try/catch", then exceptions are NOT handled there - say so explicitly.
4. Quote exact names (identifiers, types, flags, exception types) as written.
5. If the context gives only a partial answer, give what is known and add:
   "Not fully specified in the provided code."
6. If the context does not contain the answer, reply exactly:
   "Not found in the provided code."
7. Be concise (2-5 sentences). Do not output code blocks unless asked.
"""


class LLMError(RuntimeError):
    pass


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if not url.endswith("/v1") and "/v1/" not in url:
        url += "/v1"
    return url


class OpenAIChat:
    """Minimal OpenAI-compatible chat client (stdlib only)."""

    def __init__(self, base_url: str, model: str | None = None,
                 api_key: str | None = None, timeout: int = 300,
                 max_retries: int = 2, temperature: float = 0.0):
        self.base_url = normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key or ""
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.last_timings: dict = {}

    # ---- discovery --------------------------------------------------------
    def models(self) -> list[str]:
        with urllib.request.urlopen(f"{self.base_url}/models", timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("id") or m.get("model") or ""
                for m in data.get("data", []) or []]

    def health(self) -> bool:
        try:
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            with urllib.request.urlopen(f"{root}/health", timeout=8) as r:
                return r.status == 200
        except Exception:
            return False

    def resolve_model(self) -> str | None:
        if self.model:
            return self.model
        try:
            ids = [m for m in self.models() if m]
        except Exception:
            return None
        if ids:
            self.model = ids[0]
        return self.model

    # ---- the call ---------------------------------------------------------
    def chat(self, messages: list[dict], max_tokens: int = 400) -> str:
        body = {"messages": messages, "temperature": self.temperature,
                "max_tokens": max_tokens, "stream": False}
        model = self.resolve_model()
        if model:
            body["model"] = model
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps(body).encode("utf-8")

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                         data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self.last_timings = data.get("timings") or {}
                return oc.strip_thinking(data["choices"][0]["message"]["content"] or "")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last = LLMError(f"HTTP {e.code}: {detail}")
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(1 + attempt)
                    continue
                raise last
            except (urllib.error.URLError, TimeoutError, KeyError,
                    json.JSONDecodeError) as e:
                last = LLMError(f"{type(e).__name__}: {e}")
                if attempt < self.max_retries:
                    time.sleep(1 + attempt)
                    continue
                raise last
        raise last or LLMError("request failed")

    # ---- pipeline roles ---------------------------------------------------
    def draft(self, question: str, context: str) -> str:
        return self.chat([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user",
                           "content": f"Context:\n{context}\n\nQuestion: {question}"}],
                         max_tokens=400).strip()

    def verify(self, question: str, context: str, answer: str):
        return oc.parse_verdict(self.chat(
            [{"role": "user", "content": oc.build_verify_prompt(question, context, answer)}],
            max_tokens=400).strip())

    def correct(self, question: str, context: str, answer: str, critique: list[str]) -> str:
        return self.chat(
            [{"role": "user",
              "content": oc.build_correct_prompt(question, context, answer, critique)}],
            max_tokens=400).strip()


def client_from_env(base_url: str | None = None, model: str | None = None,
                    api_key: str | None = None) -> OpenAIChat | None:
    """Build a client from flags or env, or return None when unconfigured."""
    url = base_url or os.environ.get("PROD_RAG_LLM_BASE_URL")
    if not url:
        return None
    return OpenAIChat(url, model=model or os.environ.get("PROD_RAG_LLM_MODEL"),
                      api_key=api_key or os.environ.get("PROD_RAG_LLM_API_KEY"))


def format_context(idx, search: dict, max_chars: int = 6000) -> str:
    """Render a search result exactly like the diploma pipeline does."""
    lines = ["### ONTOLOGY FACTS (structured, exact):", "", search["facts"], "",
             f"### RELEVANT CODE CHUNKS (index: {idx.name}, root: {idx.root}):", ""]
    for ch in search["chunks"]:
        note = f" [{ch['file']}]" if ch.get("file") else ""
        lines.append(f"// {ch.get('kind')} {ch.get('qname')}{note} "
                     f"(lines {ch.get('line_start')}-{ch.get('line_end')})")
        lines.append(ch.get("text", ""))
        lines.append("")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + "\n... [context truncated]"
    return text


def answer_with_pipeline(idx, question: str, chat, top_k: int = 5,
                         max_chars: int = 6000) -> dict:
    """Full retrieve -> generate -> verify -> (correct) path for one question."""
    t0 = time.time()
    search = idx.search(question, top_k=top_k)
    context = format_context(idx, search, max_chars=max_chars)
    answer, vinfo = oc.verified_answer(
        question, context,
        lambda q, c: chat.draft(q, c),
        lambda q, c, a: chat.verify(q, c, a),
        lambda q, c, a, crit: chat.correct(q, c, a, crit))
    flag = ("corrected" if vinfo.get("corrected") and not vinfo.get("correction_rejected")
            else "correction-rejected" if vinfo.get("correction_rejected")
            else "supported" if vinfo.get("supported") else "unsupported")
    return {
        "index": idx.name,
        "question": question,
        "answer": answer,
        "context": context,
        "facts": search["facts"],
        "linked": search["linked"],
        "chunks": [{k: c.get(k) for k in ("id", "kind", "qname", "file",
                                          "line_start", "line_end", "text")}
                   for c in search["chunks"]],
        "verification": {**vinfo, "flag": flag},
        "timings": {"total_s": round(time.time() - t0, 2),
                    "model": getattr(chat, "last_timings", {})},
        "model": (chat.model or "").rsplit("/", 1)[-1],
    }
