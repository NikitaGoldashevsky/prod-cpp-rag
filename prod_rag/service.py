# -*- coding: utf-8 -*-
"""
service.py — one facade shared by the MCP server and the web UI.

Both front ends call this, so they can never disagree about what the codebase
contains. It owns the set of loaded indexes and merges results across them.
"""
from __future__ import annotations

import threading
import time

from . import index as idx_mod
from . import store as store_mod


class Service:
    """Loaded indexes + the operations both front ends offer."""

    def __init__(self, names: list[str] | None = None, auto_load_all: bool = True,
                 store_kind: str | None = None):
        self._lock = threading.RLock()
        self.indexes: dict[str, idx_mod.LoadedIndex] = {}
        self.kind = (store_kind or store_mod.store_kind()).lower()
        self._neo: store_mod.Neo4jStore | None = None
        if self.kind == "neo4j":
            # One driver shared by every index; the graph holds one dataset at
            # a time, so multi-index search does not apply on this backend.
            self._neo = store_mod.Neo4jStore()
        if names:
            self.indexes = self._load(names)
        elif auto_load_all:
            self.indexes = self._load(None)

    def _load(self, names):
        loaded = idx_mod.load_all(names)
        if self._neo is not None:
            for item in loaded.values():
                item._store = self._neo
        return loaded

    # ---- registry ---------------------------------------------------------
    def reload(self, names: list[str] | None = None) -> list[str]:
        with self._lock:
            self.indexes = self._load(names)
            return sorted(self.indexes)

    def names(self) -> list[str]:
        return sorted(self.indexes)

    def get(self, name: str | None) -> idx_mod.LoadedIndex:
        """Resolve an index name; with one index loaded, None means 'that one'."""
        with self._lock:
            if not self.indexes:
                raise LookupError(
                    "no indexes are loaded — run `prod-rag index add <name> <path>` first")
            if name:
                if name not in self.indexes:
                    raise LookupError(
                        f"unknown index {name!r}; loaded: {', '.join(self.names())}")
                return self.indexes[name]
            if len(self.indexes) == 1:
                return next(iter(self.indexes.values()))
            raise LookupError(
                f"several indexes are loaded ({', '.join(self.names())}); "
                "pass `index` to choose one")

    def pick(self, name: str | None) -> list[idx_mod.LoadedIndex]:
        """Indexes to search: one named, or all loaded."""
        if name:
            return [self.get(name)]
        return [self.indexes[n] for n in self.names()]

    # ---- status -----------------------------------------------------------
    def status(self, deep: bool = True) -> list[dict]:
        rows = []
        for name in self.names():
            idx = self.indexes[name]
            row = {
                "name": idx.name,
                "label": idx.label,
                "root": idx.root,
                "built_at": idx.built_at,
                "stats": idx.stats,
            }
            if deep:
                fresh = idx.freshness(rescan=True)
                row["freshness"] = {
                    "state": fresh["state"],
                    "detail": fresh["detail"],
                    "checked_at": fresh["checked_at"],
                    "changed": fresh["changed"][:50],
                    "added": fresh["added"][:50],
                    "removed": fresh["removed"][:50],
                    "folders": fresh["folders"],
                }
            rows.append(row)
        return rows

    # ---- read-only knowledge ---------------------------------------------
    def search(self, query: str, index: str | None = None, top_k: int = 8) -> dict:
        results = []
        for idx in self.pick(index):
            r = idx.search(query, top_k=top_k)
            r["index"] = idx.name
            r["root"] = idx.root
            # qualify ids so the same entity in two indexes is unambiguous
            for c in r["chunks"]:
                c["id"] = f"{idx.name}::{c['id']}"
            for e in r["linked"]:
                e["id"] = f"{idx.name}::{e['id']}"
            results.append(r)
        if len(results) == 1:
            return results[0]
        return {"query": query, "indexes": results,
                "hint": next((r["hint"] for r in results if r.get("hint")), None)}

    def entity(self, name: str, index: str | None = None) -> dict:
        found = []
        for idx in self.pick(index):
            payload = idx.entity(name)
            if payload:
                found.append(payload)
        if not found:
            return {"found": False, "name": name,
                    "searched": [i.name for i in self.pick(index)]}
        if len(found) == 1:
            return {"found": True, **found[0]}
        return {"found": True, "ambiguous": True, "matches": found}

    def neighbours(self, name: str, index: str | None = None, hops: int = 1) -> dict:
        out = []
        for idx in self.pick(index):
            r = idx.neighbours(name, hops=hops)
            if r.get("found"):
                out.append(r)
        if not out:
            return {"found": False, "name": name}
        if len(out) == 1:
            return out[0]
        return {"found": True, "matches": out}

    def find(self, pattern: str, index: str | None = None, limit: int = 50) -> dict:
        hits = []
        for idx in self.pick(index):
            for h in idx.find(pattern, limit=limit):
                h["index"] = idx.name
                hits.append(h)
        return {"pattern": pattern, "count": len(hits),
                "matches": hits[:limit * max(1, len(self.pick(index)))]}

    def chunk(self, chunk_id: str, index: str | None = None) -> dict:
        """Read one chunk. Accepts `index::chunk-id` or a bare chunk id."""
        want = chunk_id
        if "::" in chunk_id:
            pref, want = chunk_id.split("::", 1)
            cand = self.indexes.get(pref)
            if cand is not None:
                payload = cand.chunk(want)
                if payload:
                    return {"found": True, **payload}
        for idx in self.pick(index):
            payload = idx.chunk(want)
            if payload:
                payload["id"] = f"{idx.name}::{payload['id']}"
                return {"found": True, **payload}
        # Accept an entity name/qname too: agents often have one and not the id.
        for idx in self.pick(index):
            eid = idx._resolve(want)
            if eid:
                payload = idx.chunk(eid)
                if payload:
                    payload["id"] = f"{idx.name}::{payload['id']}"
                    return {"found": True, **payload}
        return {"found": False, "id": chunk_id}

    def reindex(self, name: str, verbose: bool = False) -> dict:
        """Rebuild one index from the same root with the same settings."""
        with self._lock:
            meta = next((m for m in idx_mod.list_indexes() if m["name"] == name), None)
            if meta is None:
                raise LookupError(f"unknown index {name!r}")
            new_meta = idx_mod.build(name, meta["root"], exts=tuple(meta["exts"]),
                                     exclude=tuple(meta["exclude"]),
                                     label=meta.get("label"), verbose=verbose)
            self.indexes[name] = idx_mod.load(name)
            return new_meta

    # ---- model-backed Q&A (optional) -------------------------------------
    def answer(self, question: str, chat, index: str | None = None,
               top_k: int = 5, max_chars: int = 6000) -> dict:
        if chat is None:
            raise RuntimeError(
                "no model configured — pass --llm-base-url (or set "
                "PROD_RAG_LLM_BASE_URL) to enable the Q&A box")
        idx = self.get(index)
        with self._lock:
            return idx.answer(question, chat, top_k=top_k, max_chars=max_chars)
