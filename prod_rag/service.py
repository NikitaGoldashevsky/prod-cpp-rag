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
        self._graph_cache: dict[str, dict] = {}
        if self.kind == "neo4j":
            # One driver shared by every index; each index gets a view scoped to
            # its own dataset, so several submodules can be live at once.
            self._neo = store_mod.Neo4jStore()
        if names:
            self.indexes = self._load(names)
        elif auto_load_all:
            self.indexes = self._load(None)

    def _load(self, names):
        loaded = idx_mod.load_all(names)
        if self._neo is not None:
            for item in loaded.values():
                item._store = self._neo.for_dataset(item.name)
        return loaded

    # ---- graph readiness --------------------------------------------------
    def graphs(self, refresh: bool = False) -> dict[str, dict]:
        """What the Neo4j graph holds, keyed by dataset (empty for memory)."""
        if self._neo is None:
            return {}
        if refresh or not self._graph_cache:
            self._graph_cache = {r["dataset"]: r for r in self._neo.datasets()}
        return self._graph_cache

    def _ready(self, idx: "idx_mod.LoadedIndex") -> dict | None:
        """Refuse to answer out of the wrong dataset.

        Without this, `--store neo4j` happily answered from whatever happened to
        be in the graph while `list_indexes` reported this index's file stats —
        confident, wrong, and invisible. Refusing with the fix is the honest
        behaviour.
        """
        if self._neo is None:
            return None
        info = self.graphs().get(idx.name)
        if info is None:
            info = self.graphs(refresh=True).get(idx.name)
        if info is None or not info.get("entities"):
            have = ", ".join(sorted(self._graph_cache)) or "none"
            raise store_mod.StoreNotReady(
                f"index {idx.name!r} is not in the Neo4j graph, so answering would "
                f"use the wrong data. Push it:  prod-rag neo4j push {idx.name}\n"
                f"graph holds: {have}")
        return info

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
        graphs = self.graphs()
        for name in self.names():
            idx = self.indexes[name]
            row = {
                "name": idx.name,
                "label": idx.label,
                "root": idx.root,
                "built_at": idx.built_at,
                "stats": idx.stats,
                "store": None,
            }
            if self._neo is not None:
                info = graphs.get(name)
                want = (idx.stats or {}).get("entities")
                row["store"] = {
                    "backend": "neo4j",
                    "ready": bool(info and info.get("entities")),
                    "entities": (info or {}).get("entities", 0),
                    "pushed_at": (info or {}).get("pushed_at"),
                }
                if not info:
                    row["store"]["hint"] = f"not pushed: prod-rag neo4j push {name}"
                elif want and info.get("entities", 0) != want:
                    row["store"]["hint"] = (
                        f"graph has {info.get('entities')} entities, index has {want} — "
                        f"re-push after reindexing: prod-rag neo4j push {name}")
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
            self._ready(idx)
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

    def _unprefix(self, name: str) -> str:
        """Accept `index::id`, the form search_code emits for ids."""
        if "::" in name:
            pref, rest = name.split("::", 1)
            if pref in self.indexes:
                return rest
        return name

    def entity(self, name: str, index: str | None = None) -> dict:
        found = []
        name = self._unprefix(name)
        for idx in self.pick(index):
            self._ready(idx)
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
        name = self._unprefix(name)
        for idx in self.pick(index):
            self._ready(idx)
            r = idx.neighbours(name, hops=hops)
            if r.get("found"):
                out.append(r)
        if not out:
            return {"found": False, "name": name}
        if len(out) == 1:
            return out[0]
        return {"found": True, "matches": out}

    def find(self, pattern: str, index: str | None = None, limit: int = 50) -> dict:
        if not (pattern or "").strip():
            # An empty pattern used to return the first N entities in the index,
            # which looks like a real answer. Say what is wrong instead.
            return {"pattern": pattern, "count": 0, "matches": [],
                    "error": "empty pattern", "hint": "pass a name or glob, e.g. 'Tcp*'"}
        hits = []
        totals = {}
        for idx in self.pick(index):
            self._ready(idx)
            # Ask for the full population so `total` is honest; `count` used to
            # be the page size, which understates any common name.
            found = idx.find(pattern, limit=10_000)
            totals[idx.name] = len(found)
            for h in found[:limit]:
                h["index"] = idx.name
                hits.append(h)
        total = sum(totals.values())
        out = {"pattern": pattern, "count": len(hits), "total": total,
               "matches": hits}
        if total > len(hits):
            out["truncated"] = True
            out["hint"] = (f"{total} matches, showing the first {len(hits)}; "
                           f"raise `limit` or use a narrower pattern")
        return out

    def chunk(self, chunk_id: str, index: str | None = None) -> dict:
        """Read one chunk. Accepts `index::chunk-id` or a bare chunk id."""
        want = chunk_id
        if "::" in chunk_id:
            pref, want = chunk_id.split("::", 1)
            cand = self.indexes.get(pref)
            if cand is not None:
                self._ready(cand)
                payload = cand.chunk(want)
                if payload:
                    return {"found": True, **payload}
        for idx in self.pick(index):
            self._ready(idx)
            payload = idx.chunk(want)
            if payload:
                payload["id"] = f"{idx.name}::{payload['id']}"
                return {"found": True, **payload}
        # Accept an entity name/qname too: agents often have one and not the id.
        for idx in self.pick(index):
            self._ready(idx)
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
            loaded = idx_mod.load(name)
            if self._neo is not None:
                # Rebuild and push together: a rebuilt file index whose graph
                # still holds yesterday's entities is the exact silent-wrong-
                # answer failure this backend must not have.
                loaded._store = self._neo.for_dataset(name)
                stats = self._neo.load_index(loaded.payload, clear=True)
                info = self._neo.dataset_info(name) or {}
                self._graph_cache[name] = {
                    "dataset": name, "entities": stats["entities"],
                    "chunks": stats["chunks"],
                    "pushed_at": info.get("pushed_at"),
                    "root": info.get("root")}
                new_meta = dict(new_meta, pushed=stats)
            self.indexes[name] = loaded
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
