# -*- coding: utf-8 -*-
"""
store.py — pluggable graph storage.

Two implementations behind one protocol:

  MemoryStore   the ontology held in Python (default; zero dependencies)
  Neo4jStore    the ontology held in a local Neo4j instance, queried in Cypher

The rest of PROD_RAG (index freshness, per-folder hashes, the MCP tools, the web
UI) does not care which one is active. Selection is explicit:

    PROD_RAG_STORE=memory     (default)
    PROD_RAG_STORE=neo4j      NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD

Why Neo4j is the right default at scale
---------------------------------------
With ~40k entities and ~200k relations (500 files / 185k lines), Python-side
entity linking took 400-850 ms per query because it scores every entity, and
graph expansion rescanned the relation list. Neo4j answers both with indexes and
a single Cypher statement. The store interface is the same either way, so
`tests/test_store_parity.py` can prove both backends agree.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Protocol

from . import core as oc

# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------

DEFAULT_CREDS = os.path.expanduser("~/.local/opt/neo4j/credentials.env")


def neo4j_settings(env_path: str | None = None) -> dict:
    """Resolve Neo4j connection settings from env, then a credentials file."""
    uri = os.environ.get("NEO4J_URI") or os.environ.get("PROD_RAG_NEO4J_URI")
    user = os.environ.get("NEO4J_USER") or os.environ.get("PROD_RAG_NEO4J_USER")
    pw = os.environ.get("NEO4J_PASSWORD") or os.environ.get("PROD_RAG_NEO4J_PASSWORD")
    path = env_path or os.environ.get("NEO4J_ENV") or DEFAULT_CREDS
    if not (uri and pw) and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            k, _, v = line.strip().partition("=")
            if k == "NEO4J_URI":
                uri = uri or v
            elif k in ("NEO4J_USER", "NEO4J_USERNAME"):
                user = user or v
            elif k == "NEO4J_PASSWORD":
                pw = pw or v
    if not (uri and pw):
        raise RuntimeError(
            "Neo4j is not configured. Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD "
            f"or create {DEFAULT_CREDS}. See README §Neo4j.")
    return {"uri": uri, "user": user or "neo4j", "password": pw}


def store_kind() -> str:
    return (os.environ.get("PROD_RAG_STORE") or "memory").strip().lower()


class StoreNotReady(RuntimeError):
    """The graph cannot answer for this index (not pushed, or pushed empty).

    Raised instead of returning a confidently wrong answer. The message always
    names the command that fixes it.
    """


# ---------------------------------------------------------------------------
# Symbol matching (shared by both backends so they cannot drift)
# ---------------------------------------------------------------------------

_CAMEL_BREAK = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _word_starts(text: str) -> set[int]:
    """Offsets where a new word begins: after ::, _, /, or a CamelCase hump."""
    starts = {0}
    for i, ch in enumerate(text):
        if ch in "::_/ " or ch in ".-><":
            starts.add(i + 1)
    for m in _CAMEL_BREAK.finditer(text):
        starts.add(m.start())
    return starts


def match_rank(qname: str, name: str, pattern: str) -> tuple[int, str]:
    """How well a symbol matches, best first.

    Substring search is what users expect, but it is also what makes
    `find_symbol("Tcp")` return `adjustCpu` (the letters *tcp* really are in
    there). So hits are still returned, but ranked: a match that starts on a
    word boundary beats one buried mid-word, and each hit says which it is.
    """
    if any(ch in pattern for ch in "*?[]"):
        return 0, "glob"
    p = pattern.lower()
    best: tuple[int, str] | None = None
    for text in (qname, name):
        if not text:
            continue
        low = text.lower()
        if low == p:
            return 0, "exact"
        idx = low.find(p)
        if idx < 0:
            continue
        cand = (1, "prefix") if idx == 0 else \
               (2, "word") if idx in _word_starts(text) else (3, "substring")
        if best is None or cand[0] < best[0]:
            best = cand
    return best or (3, "substring")


def class_location(declared_line: int, members: list[dict]) -> dict:
    """Honest location for a class/struct.

    A class seen only through `Class::method` definitions has no declaration in
    the index (the header lives outside the indexed subtree). Reporting
    `line_start: 0` silently is what makes a tool look broken; instead say the
    declaration is absent and point at the first member that *is* in the tree.
    """
    if declared_line:
        return {"declared_in_index": True, "first_member": None}
    tops = sorted((m for m in members if m.get("line_start")),
                  key=lambda m: (m.get("file") or "", m.get("line_start") or 0))
    return {"declared_in_index": False,
            "first_member": ({"qname": tops[0].get("qname"),
                              "file": tops[0].get("file"),
                              "line_start": tops[0].get("line_start")}
                             if tops else None)}


def rank_symbol_hits(hits: list[dict], pattern: str, limit: int) -> list[dict]:
    """Sort hits word-boundary-first and annotate each with its `match` kind."""
    out = []
    for h in hits:
        qn = h.get("qname") or ""
        nm = h.get("name") or qn.rsplit("::", 1)[-1]
        rank, kind = match_rank(qn, nm, pattern)
        out.append({**h, "_rank": rank, "match": kind})
    out.sort(key=lambda h: (h["_rank"], len(h.get("qname") or ""),
                            h.get("qname") or ""))
    drop = ("_rank", "name_lc", "qname_lc")
    return [{k: v for k, v in h.items() if k not in drop} for h in out[:limit]]


# ---------------------------------------------------------------------------
# Protocol: what every backend must provide
# ---------------------------------------------------------------------------

class GraphStore(Protocol):
    name: str

    def search(self, query: str, top_k: int = 8, hops: int = 1) -> dict: ...
    def entity(self, name: str) -> dict | None: ...
    def neighbours(self, name: str, hops: int = 1) -> dict: ...
    def find(self, pattern: str, limit: int = 50) -> list[dict]: ...
    def chunk(self, chunk_id: str) -> dict | None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Memory backend
# ---------------------------------------------------------------------------

class MemoryStore:
    """The in-memory ontology (the original behaviour, unchanged)."""

    name = "memory"

    def __init__(self, onto: oc.CppOntology):
        self.onto = onto

    # -- search ------------------------------------------------------------
    def search(self, query: str, top_k: int = 8, hops: int = 1) -> dict:
        onto = self.onto
        res = onto.retrieve(query, top_k=top_k, hops=hops)
        facts = res["facts"]
        if not facts:
            cards, seen = [], set()
            for ch, _ in res["chunks"][:3]:
                eid = ch["id"]
                if eid in onto.entities and eid not in seen:
                    seen.add(eid)
                    cards.append(onto._card(onto.entities[eid]))
            facts = "\n\n".join(cards)
        return {
            "linked": [
                {"id": eid,
                 "name": onto.entities[eid].name,
                 "qname": onto.entities[eid].qname or onto.entities[eid].name,
                 "kind": onto.entities[eid].kind,
                 "file": onto.entities[eid].file,
                 "line_start": onto.entities[eid].line_start,
                 "line_end": onto.entities[eid].line_end}
                for eid in res["linked"] if eid in onto.entities],
            "facts": facts,
            "hint": (None if res["chunks"] else
                     "no chunk matched this wording — try a symbol name from the "
                     "codebase, or use find_symbol/glob (e.g. 'Tcp*')"),
            "chunks": [
                {"id": ch["id"], "kind": ch.get("kind"), "qname": ch.get("qname"),
                 "file": ch.get("file"), "line_start": ch.get("line_start"),
                 "line_end": ch.get("line_end"), "text": ch.get("text", "")}
                for ch, _ in res["chunks"]],
        }

    # -- lookups -----------------------------------------------------------
    def _resolve(self, name: str) -> str | None:
        onto = self.onto
        if name in onto.entities:
            return name
        low = name.lower()
        exact = [eid for eid, e in onto.entities.items()
                 if (e.qname or "").lower() == low or e.name.lower() == low]
        if exact:
            exact.sort(key=lambda eid: (not onto.entities[eid].is_definition,
                                        onto.entities[eid].kind in ("Parameter", "File")))
            return exact[0]
        # Fuzzy fallback: never a file/include stub. `entity("NvDateTime")`
        # used to resolve to the `#include "nvdatetime.h"` Header node and
        # report `kind: Header, file: ""` -- a "found" that answers nothing.
        stub_kinds = ("Header", "File", "GlobalBlock", "Parameter")
        sub = [eid for eid, e in onto.entities.items()
               if low in (e.qname or e.name).lower()
               and e.kind not in stub_kinds]
        if sub:
            sub.sort(key=lambda eid: (len(onto.entities[eid].qname
                                         or onto.entities[eid].name),
                                      onto.entities[eid].kind != "Class"))
            return sub[0]
        return None

    def entity(self, name: str) -> dict | None:
        onto = self.onto
        eid = self._resolve(name)
        if eid is None:
            return None
        e = onto.entities[eid]
        lab = lambda x: (onto.entities[x].qname if x in onto.entities else x)
        chunk = next((c for c in onto.chunks if c["id"] == eid), None)
        loc = {}
        if e.kind in ("Class", "Struct"):
            kids = [onto.entities[i] for i in
                    onto._out(eid, "hasMethod") + onto._out(eid, "hasConstructor")
                    + onto._out(eid, "hasField") if i in onto.entities]
            loc = class_location(e.line_start, [
                {"qname": k.qname or k.name, "file": k.file,
                 "line_start": k.line_start} for k in kids])
        return {
            "id": eid, "kind": e.kind, "name": e.name, "qname": e.qname or e.name,
            "file": e.file, "line_start": e.line_start, "line_end": e.line_end,
            "is_definition": e.is_definition, "return_type": e.return_type,
            "type_ref": e.type_ref, "params": e.params,
            "modifiers": [k for k, v in
                          (("const", e.is_const), ("static", e.is_static),
                           ("virtual", e.is_virtual), ("noexcept", e.is_noexcept)) if v],
            "has_try_catch": e.has_try_catch, "throw_sites": e.throw_sites,
            "calls": sorted({lab(o) for o in onto._out(eid, "calls")}),
            "callers": sorted({lab(r.s) for r in onto.relations
                               if r.o == eid and r.p == "calls"}),
            "references": sorted({lab(o) for o in onto._out(eid, "references")}),
            "uses": sorted(set(onto._out(eid, "callsExternal"))),
            "initializes": [lab(o) for o in onto._out(eid, "initializes")],
            "inherits": onto._out(eid, "inheritsFrom"),
            "inherited_by": [r.s for r in onto.relations
                             if r.o == eid and r.p == "inheritsFrom"],
            "members": sorted({lab(o) for o in
                               onto._out(eid, "hasMethod") +
                               onto._out(eid, "hasConstructor") +
                               onto._out(eid, "hasField")}),
            **loc,
            "fact_card": onto._card(e),
            "code": chunk["text"] if chunk else e.code,
        }

    def neighbours(self, name: str, hops: int = 1) -> dict:
        onto = self.onto
        eid = self._resolve(name)
        if eid is None:
            return {"found": False, "name": name}
        ids = onto.neighbour_ids([eid], hops=max(1, min(hops, 3)))
        return {
            "found": True, "root": eid,
            "entities": [{"id": i, "kind": onto.entities[i].kind,
                          "qname": onto.entities[i].qname or onto.entities[i].name,
                          "file": onto.entities[i].file}
                         for i in ids if i in onto.entities],
            "facts": onto.fact_cards(ids),
        }

    def find(self, pattern: str, limit: int = 50) -> list[dict]:
        import fnmatch
        onto = self.onto
        pat = pattern.lower()
        globbed = any(ch in pattern for ch in "*?[]")
        hits = []
        for eid, e in onto.entities.items():
            qn = e.qname or e.name
            target = qn.lower()
            ok = (fnmatch.fnmatch(target, pat) or fnmatch.fnmatch(e.name.lower(), pat)) \
                if globbed else (pat in target or pat in e.name.lower())
            if ok and e.kind not in ("Parameter", "File", "GlobalBlock", "Header"):
                hits.append({"id": eid, "kind": e.kind, "qname": qn,
                             "name": e.name,
                             "file": e.file, "line_start": e.line_start,
                             "is_definition": e.is_definition})
        hits.sort(key=lambda h: (len(h["qname"]), h["qname"]))
        return rank_symbol_hits(hits, pattern, limit)

    def chunk(self, chunk_id: str) -> dict | None:
        return next((c for c in self.onto.chunks if c["id"] == chunk_id), None)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Neo4j backend
# ---------------------------------------------------------------------------

KIND_LABELS = ("File", "Header", "Class", "Struct", "Method", "Function",
               "Constructor", "Destructor", "Field", "Parameter", "TypeAlias",
               "GlobalBlock")

REL_TYPES = {
    "includes": "INCLUDES", "declares": "DECLARES", "hasMethod": "HAS_METHOD",
    "hasConstructor": "HAS_CONSTRUCTOR", "hasField": "HAS_FIELD",
    "hasParameter": "HAS_PARAMETER", "calls": "CALLS", "references": "REFERENCES",
    "initializes": "INITIALIZES",
}
REL_BY_TYPE = {v: k for k, v in REL_TYPES.items()}
LITERAL_PREDICATES = {"inheritsFrom", "callsExternal", "throws", "aliasOf"}
ENTITY_REL_TYPES = [t for p, t in REL_TYPES.items() if p not in LITERAL_PREDICATES]

SCHEMA = [
    # Composite uniqueness: the same entity id may exist in several datasets
    # (one per indexed submodule), so `id` alone is no longer a key.
    "DROP CONSTRAINT entity_id IF EXISTS",
    "DROP CONSTRAINT chunk_id IF EXISTS",
    "CREATE CONSTRAINT entity_dataset_id IF NOT EXISTS "
    "FOR (n:Entity) REQUIRE (n.dataset, n.id) IS UNIQUE",
    "CREATE CONSTRAINT chunk_dataset_id IF NOT EXISTS "
    "FOR (n:Chunk) REQUIRE (n.dataset, n.id) IS UNIQUE",
    "CREATE CONSTRAINT dataset_name IF NOT EXISTS "
    "FOR (n:Dataset) REQUIRE n.name IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    "CREATE INDEX entity_qname IF NOT EXISTS FOR (n:Entity) ON (n.qname)",
    "CREATE INDEX entity_qname_lc IF NOT EXISTS FOR (n:Entity) ON (n.qname_lc)",
    "CREATE INDEX entity_kind IF NOT EXISTS FOR (n:Entity) ON (n.kind)",
    "CREATE INDEX entity_file IF NOT EXISTS FOR (n:Entity) ON (n.file)",
    "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS "
    "FOR (n:Entity) ON EACH [n.name, n.qname, n.code, n.fact_card]",
    "CREATE FULLTEXT INDEX chunk_text IF NOT EXISTS FOR (n:Chunk) ON EACH [n.text, n.qname]",
]


class Neo4jStore:
    """The ontology in Neo4j. Query paths are Cypher; ranking is index-driven.

    Every node carries a `dataset` property naming the index it came from, and
    every query filters on it. That is what lets one Neo4j instance hold several
    submodules at once — and, just as important, what stops a query from
    answering out of a *different* index than the one the caller asked for.
    """

    name = "neo4j"
    MAX_NEIGHBOURS = 300

    def __init__(self, dataset: str | None = None, database: str = "neo4j",
                 env_path: str | None = None, driver=None, uri: str | None = None):
        self.dataset = dataset
        self.database = database
        self._owns_driver = driver is None
        if driver is None:
            from neo4j import GraphDatabase
            c = neo4j_settings(env_path)
            self.uri = uri or c["uri"]
            driver = GraphDatabase.driver(
                self.uri, auth=(c["user"], c["password"]),
                warn_notification_severity="OFF", max_connection_pool_size=16)
            driver.verify_connectivity()
        else:
            self.uri = uri or "neo4j"
        self.driver = driver

    # ---- helpers ---------------------------------------------------------
    def for_dataset(self, dataset: str) -> "Neo4jStore":
        """A view of this connection scoped to one dataset (shares the driver)."""
        return Neo4jStore(dataset=dataset, database=self.database,
                          driver=self.driver, uri=self.uri)

    def _ds(self) -> str:
        if not self.dataset:
            raise StoreNotReady(
                "this Neo4j handle is not bound to a dataset; use for_dataset(name)")
        return self.dataset

    def _session(self):
        return self.driver.session(database=self.database,
                                   notifications_min_severity="OFF")

    def ensure_schema(self) -> None:
        with self._session() as s:
            for stmt in SCHEMA:
                s.run(stmt).consume()

    @staticmethod
    def _batch(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    # ---- dataset state ---------------------------------------------------
    def dataset_info(self, dataset: str | None = None) -> dict | None:
        """What the graph holds for one dataset, or None if it holds nothing."""
        ds = dataset or self.dataset
        if not ds:
            raise ValueError("dataset_info() needs a dataset name")
        with self._session() as s:
            rec = s.run(
                "MATCH (d:Dataset {name:$ds}) "
                "RETURN d.name AS name, d.pushed_at AS pushed_at, "
                "       d.index_built_at AS index_built_at, d.root AS root",
                ds=ds).single()
            counts = s.run(
                "MATCH (e:Entity {dataset:$ds}) WITH count(e) AS entities "
                "CALL { WITH entities MATCH (c:Chunk {dataset:$ds}) "
                "       RETURN count(c) AS chunks } "
                "RETURN entities, chunks", ds=ds).single()
        if not rec and not counts["entities"]:
            return None
        return {"dataset": ds,
                "entities": counts["entities"],
                "chunks": counts["chunks"],
                "pushed_at": rec["pushed_at"] if rec else None,
                "index_built_at": rec["index_built_at"] if rec else None,
                "root": rec["root"] if rec else None}

    def datasets(self) -> list[dict]:
        """Every dataset in the graph, with the counts each query path sees."""
        stmts = (
            "MATCH (e:Entity) WHERE e.dataset IS NOT NULL "
            "WITH e.dataset AS dataset, count(e) AS entities "
            "CALL { WITH dataset MATCH (c:Chunk {dataset: dataset}) "
            "       RETURN count(c) AS chunks } "
            "RETURN dataset, entities, chunks ORDER BY dataset")
        with self._session() as s:
            rows = s.run(stmts).data()
            meta = {r["name"]: r for r in s.run(
                "MATCH (d:Dataset) RETURN d.name AS name, d.pushed_at AS pushed_at, "
                "d.index_built_at AS index_built_at, d.root AS root").data()}
        out = []
        for r in rows:
            m = meta.get(r["dataset"], {})
            out.append({"dataset": r["dataset"], "entities": r["entities"],
                        "chunks": r["chunks"],
                        "pushed_at": m.get("pushed_at"),
                        "index_built_at": m.get("index_built_at"),
                        "root": m.get("root")})
        return out

    def assert_ready(self) -> dict:
        """Raise StoreNotReady (with a fix) unless this dataset is in the graph."""
        info = self.dataset_info()
        if info is None or not info["entities"]:
            raise StoreNotReady(
                f"index {self.dataset!r} is not in the Neo4j graph at {self.uri}. "
                f"Push it first:  prod-rag neo4j push {self.dataset}  "
                f"(or rebuild:  prod-rag index add {self.dataset} <path> --store neo4j). "
                f"Graph datasets: "
                f"{', '.join(d['dataset'] for d in self.datasets()) or 'none'}")
        return info

    # ---- loading an index ------------------------------------------------
    def load_index(self, payload: dict, clear: bool = True) -> dict:
        """Write one index snapshot into its own dataset."""
        ds = payload.get("name") or self.dataset
        if not ds:
            raise ValueError("payload has no index name")
        self.ensure_schema()
        stats = {"dataset": ds, "entities": 0, "chunks": 0, "relations": 0}
        with self._session() as s:
            if clear:
                # Only this dataset: other submodules stay queryable.
                s.run(
                    "MATCH (n) WHERE n.dataset = $ds "
                    "CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS",
                    ds=ds).consume()
                s.run("MATCH (d:Dataset {name:$ds}) DETACH DELETE d", ds=ds).consume()

            # entity nodes, one statement per kind label
            all_labels = "".join(f":{x}" for x in KIND_LABELS)
            by_label: dict[str, list[dict]] = {}
            for i, e in enumerate(payload["entities"]):
                label = e["kind"] if e["kind"] in KIND_LABELS else None
                if label is None:
                    continue
                props = {"id": e["id"], "dataset": ds, "kind": e["kind"],
                         "name": e.get("name") or "",
                         "qname": e.get("qname") or "",
                         "qname_lc": (e.get("qname") or e.get("name") or "").lower(),
                         "name_lc": (e.get("name") or "").lower(),
                         "ord": i,
                         "file": e.get("file") or "",
                         "line_start": e.get("line_start") or 0,
                         "line_end": e.get("line_end") or 0,
                         "is_definition": bool(e.get("is_definition", True)),
                         "has_try_catch": bool(e.get("has_try_catch", False)),
                         "params_json": json.dumps(e.get("params") or []),
                         "throw_json": json.dumps(e.get("throw_sites") or []),
                         "fact_card": e.get("fact_card") or ""}
                for k in ("return_type", "type_ref", "parent", "code", "access"):
                    if e.get(k):
                        props[k] = e[k]
                for k in ("is_const", "is_static", "is_virtual", "is_noexcept"):
                    if e.get(k):
                        props[k] = True
                by_label.setdefault(label, []).append({"id": e["id"], "props": props})

            for label, rows in by_label.items():
                for chunk in self._batch(rows, 1000):
                    s.run(f"UNWIND $rows AS row "
                          f"MERGE (n:Entity {{dataset: $ds, id: row.id}}) "
                          f"REMOVE n{all_labels} "
                          f"SET n = row.props SET n:{label}", rows=chunk,
                          ds=ds).consume()
                    stats["entities"] += len(chunk)

            # relationships (multiset preserved via `ords`)
            grouped: dict[tuple, list[int]] = {}
            for i, r in enumerate(payload["relations"]):
                if r["p"] in LITERAL_PREDICATES:
                    continue
                rtype = REL_TYPES.get(r["p"])
                if rtype is None:
                    continue
                grouped.setdefault((rtype, r["s"], r["o"]), []).append(i)
            by_type: dict[str, list[dict]] = {}
            for (rtype, subj, obj), ords in grouped.items():
                by_type.setdefault(rtype, []).append({"s": subj, "o": obj, "ords": ords})
            for rtype, rows in by_type.items():
                for chunk in self._batch(rows, 5000):
                    s.run(f"UNWIND $rows AS row "
                          f"MATCH (a:Entity {{dataset: $ds, id: row.s}}) "
                          f"MATCH (b:Entity {{dataset: $ds, id: row.o}}) "
                          f"MERGE (a)-[r:{rtype}]->(b) SET r.ords = row.ords",
                          rows=chunk, ds=ds).consume()
                    stats["relations"] += sum(len(r["ords"]) for r in chunk)

            # literal objects -> stub nodes (scoped to the dataset too)
            for pred, rel, label, key in (
                    ("callsExternal", "CALLS_EXTERNAL", "ExternalSymbol", "name"),
                    ("inheritsFrom", "INHERITS_FROM", "ExternalBase", "name")):
                g: dict[tuple, list[int]] = {}
                for i, r in enumerate(payload["relations"]):
                    if r["p"] == pred:
                        g.setdefault((r["s"], r["o"]), []).append(i)
                rows = [{"s": a, "o": b, "ords": o} for (a, b), o in g.items()]
                for chunk in self._batch(rows, 5000):
                    s.run(f"UNWIND $rows AS row "
                          f"MATCH (a:Entity {{dataset: $ds, id: row.s}}) "
                          f"MERGE (b:{label} {{dataset: $ds, {key}: row.o}}) "
                          f"MERGE (a)-[r:{rel}]->(b) SET r.ords = row.ords",
                          rows=chunk, ds=ds).consume()
                    stats["relations"] += sum(len(r["ords"]) for r in chunk)

            # throws (guarded flag on the edge). Site details live on the
            # entity payload (throw_sites), so pair each `throws` relation with
            # the next unused site of the same type for its subject.
            sites_by_entity: dict[str, list[dict]] = {
                e["id"]: (e.get("throw_sites") or []) for e in payload["entities"]}
            cursor: dict[tuple[str, str], int] = {}
            tg: dict[tuple, list[int]] = {}
            for i, r in enumerate(payload["relations"]):
                if r["p"] != "throws":
                    continue
                key = (r["s"], r["o"])
                k = cursor.get(key, 0)
                sites = [x for x in sites_by_entity.get(r["s"], [])
                         if x.get("type") == r["o"]]
                site = sites[k] if k < len(sites) else {"guarded": False, "line": 0}
                cursor[key] = k + 1
                tg.setdefault((r["s"], r["o"], bool(site.get("guarded")),
                               int(site.get("line") or 0)), []).append(i)
            rows = [{"s": a, "o": b, "guarded": g_, "line": ln, "ords": o}
                    for (a, b, g_, ln), o in tg.items()]
            for chunk in self._batch(rows, 5000):
                s.run("UNWIND $rows AS row "
                      "MATCH (a:Entity {dataset: $ds, id: row.s}) "
                      "MERGE (b:ThrowType {dataset: $ds, name: row.o}) "
                      "MERGE (a)-[t:THROWS {line: row.line, guarded: row.guarded}]->(b) "
                      "SET t.ords = row.ords", rows=chunk, ds=ds).consume()
                stats["relations"] += sum(len(r["ords"]) for r in chunk)

            # chunks
            rows = [{"id": c["id"], "kind": c.get("kind") or "",
                     "qname": c.get("qname") or "",
                     "file": c.get("file") or "", "line_start": c.get("line_start") or 0,
                     "line_end": c.get("line_end") or 0, "text": c.get("text") or "",
                     "ord": i}
                    for i, c in enumerate(payload["chunks"])]
            for chunk in self._batch(rows, 200):
                s.run("UNWIND $rows AS row "
                      "MERGE (c:Chunk {dataset: $ds, id: row.id}) "
                      "SET c = row, c.dataset = $ds "
                      "WITH c, row MATCH (e:Entity {dataset: $ds, id: row.id}) "
                      "MERGE (c)-[:DESCRIBES]->(e)", rows=chunk, ds=ds).consume()
                stats["chunks"] += len(chunk)

            # record the dataset itself so `list_indexes` can show what the
            # graph holds and when it was pushed
            s.run(
                "MERGE (d:Dataset {name:$ds}) "
                "SET d.pushed_at = $pushed, d.index_built_at = $built, d.root = $root, "
                "    d.entities = $entities, d.chunks = $chunks, d.relations = $relations",
                ds=ds, pushed=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                built=payload.get("updated_at") or payload.get("created_at"),
                root=payload.get("root") or "",
                entities=stats["entities"], chunks=stats["chunks"],
                relations=stats["relations"]).consume()
        return stats

    def drop_dataset(self, dataset: str) -> bool:
        """Remove one dataset (its nodes, chunks and stubs) from the graph."""
        with self._session() as s:
            got = s.run("MATCH (d:Dataset {name:$ds}) RETURN count(d) AS n",
                        ds=dataset).single()["n"]
            s.run("MATCH (n) WHERE n.dataset = $ds "
                  "CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS",
                  ds=dataset).consume()
            s.run("MATCH (d:Dataset {name:$ds}) DETACH DELETE d", ds=dataset).consume()
        return bool(got)

    def clear_all(self) -> None:
        """Wipe every dataset — including nodes pushed before datasets existed."""
        with self._session() as s:
            for lab in list(KIND_LABELS) + ["Entity", "Chunk", "ExternalSymbol",
                                            "ExternalBase", "ThrowType"]:
                s.run(f"MATCH (n:{lab}) "
                      "CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS").consume()
            s.run("MATCH (d:Dataset) DETACH DELETE d").consume()

    # ---- queries ---------------------------------------------------------
    def search(self, query: str, top_k: int = 8, hops: int = 1) -> dict:
        """Full-text over entities and chunks, then one hop for surrounding facts."""
        ds = self._ds()
        like = _fts_query(query)
        linked, chunks = [], []
        with self._session() as s:
            if like:
                # The full-text index streams in score order, so the dataset
                # filter costs nothing extra: LIMIT stops the scan.
                for rec in s.run(
                        "CALL db.index.fulltext.queryNodes('entity_text', $q) "
                        "YIELD node, score "
                        "WHERE node:Entity AND node.dataset = $ds "
                        "AND NOT node:Parameter AND NOT node:File "
                        "AND NOT node:GlobalBlock AND NOT node:Header "
                        "RETURN node.id AS id, node.name AS name, node.qname AS qname, "
                        "node.kind AS kind, node.file AS file, node.line_start AS ls, "
                        "node.line_end AS le, score ORDER BY score DESC LIMIT $k",
                        q=like, ds=ds, k=max(3, top_k)):
                    if rec["id"] not in {x["id"] for x in linked}:
                        linked.append({"id": rec["id"], "name": rec["name"],
                                       "qname": rec["qname"] or rec["name"],
                                       "kind": rec["kind"], "file": rec["file"],
                                       "line_start": rec["ls"], "line_end": rec["le"]})
                for rec in s.run(
                        "CALL db.index.fulltext.queryNodes('chunk_text', $q) "
                        "YIELD node, score "
                        "WHERE node.dataset = $ds "
                        "RETURN node.id AS id, node.kind AS kind, node.qname AS qname, "
                        "node.file AS file, node.line_start AS ls, node.line_end AS le, "
                        "node.text AS text, score ORDER BY score DESC LIMIT $k",
                        q=like, ds=ds, k=top_k):
                    chunks.append({"id": rec["id"], "kind": rec["kind"],
                                   "qname": rec["qname"], "file": rec["file"],
                                   "line_start": rec["ls"], "line_end": rec["le"],
                                   "text": rec["text"]})

            facts = self._facts_for([x["id"] for x in linked][:8], hops=hops)

            if not chunks:
                seen = set()
                for rec in s.run(
                        "MATCH (c:Chunk {dataset:$ds})-[:DESCRIBES]->(e:Entity {dataset:$ds}) "
                        "WHERE e.id IN $ids RETURN c.id AS id, c.kind AS kind, "
                        "c.qname AS qname, c.file AS file, c.line_start AS ls, "
                        "c.line_end AS le, c.text AS text LIMIT $k",
                        ds=ds, ids=[x["id"] for x in linked][:8], k=top_k):
                    if rec["id"] in seen:
                        continue
                    seen.add(rec["id"])
                    chunks.append({"id": rec["id"], "kind": rec["kind"],
                                   "qname": rec["qname"], "file": rec["file"],
                                   "line_start": rec["ls"], "line_end": rec["le"],
                                   "text": rec["text"]})

        hint = None
        if not chunks:
            hint = ("no chunk matched this wording — try a symbol name from the "
                    "codebase, or use find_symbol/glob (e.g. 'Tcp*')")
        return {"linked": linked, "facts": facts, "chunks": chunks, "hint": hint}

    def _facts_for(self, ids: list[str], hops: int = 1) -> str:
        if not ids:
            return ""
        ds = self._ds()
        h = max(1, min(hops, 3))
        with self._session() as s:
            rec = s.run(
                f"MATCH (e:Entity {{dataset:$ds}}) WHERE e.id IN $ids "
                f"OPTIONAL MATCH (e)-[*1..{h}]-(n:Entity) "
                f"WITH collect(DISTINCT e) + collect(DISTINCT n) AS nodes "
                f"UNWIND nodes AS x "
                f"WITH DISTINCT x WHERE x.dataset = $ds "
                f"AND x.fact_card IS NOT NULL AND x.fact_card <> '' "
                f"AND NOT x:File AND NOT x:GlobalBlock "
                f"RETURN x.fact_card AS card, x.ord AS ord ORDER BY ord",
                ids=ids, ds=ds).data()
        return "\n\n".join(r["card"] for r in rec)

    def entity(self, name: str) -> dict | None:
        ds = self._ds()
        with self._session() as s:
            rec = s.run(
                "MATCH (e:Entity {dataset:$ds}) "
                "WHERE e.qname = $n OR e.name = $n "
                "   OR e.qname_lc = $lc OR e.name_lc = $lc OR e.id = $n "
                "RETURN e ORDER BY coalesce(e.is_definition,true) DESC, "
                "  size(coalesce(e.qname,'')) ASC LIMIT 1",
                n=name, lc=name.lower(), ds=ds).single()
            if not rec:
                # Same rule as MemoryStore: an include/file stub is never a
                # useful answer to "tell me about this symbol".
                rec = s.run(
                    "MATCH (e:Entity {dataset:$ds}) "
                    "WHERE (e.qname_lc CONTAINS $lc OR e.name_lc CONTAINS $lc) "
                    "AND NOT e:Header AND NOT e:File "
                    "AND NOT e:GlobalBlock AND NOT e:Parameter "
                    "RETURN e ORDER BY size(coalesce(e.qname,'')) ASC LIMIT 1",
                    lc=name.lower(), ds=ds).single()
            if not rec:
                return None
            node = dict(rec["e"])
            eid = node["id"]
            rel = s.run(
                "MATCH (a:Entity {dataset:$ds, id:$id})-[r]->(b) "
                "RETURN type(r) AS t, b.id AS o, b.qname AS oq, b.name AS oname, "
                "       b.file AS of, b.line_start AS ol, "
                "       r.guarded AS guarded, r.line AS line, r.ords AS ords",
                id=eid, ds=ds).data()
            inbound = s.run(
                "MATCH (a:Entity {dataset:$ds})-[r]->(b:Entity {dataset:$ds, id:$id}) "
                "RETURN type(r) AS t, a.id AS s, a.qname AS sq, a.name AS sname",
                id=eid, ds=ds).data()
            code = s.run("MATCH (c:Chunk {dataset:$ds, id:$id}) RETURN c.text AS text",
                         id=eid, ds=ds).single()
        label = lambda q, nm: q or nm
        def objs(t):
            return [label(r["oq"], r["oname"]) for r in rel if r["t"] == t]
        loc = {}
        if node.get("kind") in ("Class", "Struct"):
            kids = [{"qname": label(r["oq"], r["oname"]), "file": r["of"],
                     "line_start": r["ol"]}
                    for r in rel if r["t"] in ("HAS_METHOD", "HAS_CONSTRUCTOR",
                                               "HAS_FIELD")]
            loc = class_location(node.get("line_start") or 0, kids)
        return {
            "id": eid,
            "kind": node.get("kind"),
            "name": node.get("name"),
            "qname": node.get("qname") or node.get("name"),
            "file": node.get("file"),
            "line_start": node.get("line_start"),
            "line_end": node.get("line_end"),
            "is_definition": bool(node.get("is_definition", True)),
            "return_type": node.get("return_type"),
            "type_ref": node.get("type_ref"),
            "params": json.loads(node.get("params_json") or "[]"),
            "modifiers": [k for k in ("is_const", "is_static", "is_virtual", "is_noexcept")
                          if node.get(k)],
            "has_try_catch": bool(node.get("has_try_catch", False)),
            "throw_sites": json.loads(node.get("throw_json") or "[]"),
            "calls": sorted(set(objs("CALLS"))),
            "callers": sorted({label(r["sq"], r["sname"]) for r in inbound
                               if r["t"] == "CALLS"}),
            "references": sorted(set(objs("REFERENCES"))),
            "uses": sorted(set(objs("CALLS_EXTERNAL"))),
            "initializes": objs("INITIALIZES"),
            "inherits": sorted(set(objs("INHERITS_FROM"))),
            "inherited_by": sorted({r["s"] for r in inbound if r["t"] == "INHERITS_FROM"}),
            "members": sorted(set(objs("HAS_METHOD") + objs("HAS_CONSTRUCTOR")
                                  + objs("HAS_FIELD"))),
            **loc,
            "fact_card": node.get("fact_card") or "",
            "code": (code["text"] if code else node.get("code") or ""),
        }

    def neighbours(self, name: str, hops: int = 1) -> dict:
        ds = self._ds()
        h = max(1, min(hops, 3))
        with self._session() as s:
            # Same preference order as MemoryStore._resolve: exact qname, then
            # a definition, then the shortest name. Without ORDER BY, duplicate
            # class names across directories resolved to different nodes.
            root = s.run(
                "MATCH (e:Entity {dataset:$ds}) "
                "WHERE e.qname = $n OR e.name = $n OR e.id = $n "
                "   OR e.qname_lc = $lc OR e.name_lc = $lc "
                "RETURN e.id AS id, "
                "  CASE WHEN e.qname = $n THEN 0 WHEN e.name = $n THEN 1 "
                "       WHEN e.id = $n THEN 2 ELSE 3 END AS rank, "
                "  coalesce(e.is_definition, true) AS def, "
                "  size(coalesce(e.qname, e.name)) AS namelen, e.ord AS ord "
                "ORDER BY rank, def DESC, namelen ASC, ord ASC LIMIT 1",
                n=name, lc=name.lower(), ds=ds).single()
            if not root:
                return {"found": False, "name": name}
            rid = root["id"]
            # Cap the expansion: a well-connected class two hops out can reach
            # tens of thousands of nodes, which is useless context and slow.
            rows = s.run(
                f"MATCH (root:Entity {{dataset:$ds, id:$id}}) "
                f"OPTIONAL MATCH (root)-[*1..{h}]-(n:Entity) "
                f"WITH root, collect(DISTINCT n) AS ns "
                f"UNWIND [root] + ns[0..$cap] AS x "
                f"RETURN DISTINCT x.id AS id, x.kind AS kind, "
                f"       coalesce(x.qname, x.name) AS qname, x.file AS file, "
                f"       x.fact_card AS card, x.ord AS ord, "
                f"       (x:File OR x:GlobalBlock) AS skip "
                f"ORDER BY ord", id=rid, ds=ds, cap=self.MAX_NEIGHBOURS).data()
            total = s.run(
                f"MATCH (root:Entity {{dataset:$ds, id:$id}}) "
                f"OPTIONAL MATCH (root)-[*1..{h}]-(n:Entity) "
                f"RETURN count(DISTINCT n) AS n", id=rid, ds=ds).single()["n"]
        entities = [{"id": r["id"], "kind": r["kind"], "qname": r["qname"],
                     "file": r["file"]} for r in rows]
        # MemoryStore.fact_cards() skips File and GlobalBlock nodes; mirror it
        facts = "\n\n".join(r["card"] for r in rows if r["card"] and not r.get("skip"))
        out = {"found": True, "root": rid, "entities": entities, "facts": facts}
        if total > self.MAX_NEIGHBOURS:
            out["truncated"] = True
            out["total_neighbours"] = total
            out["note"] = (f"{total} nodes within {h} hop(s); showing the first "
                           f"{self.MAX_NEIGHBOURS}. Name a symbol or reduce hops.")
        return out

    def find(self, pattern: str, limit: int = 50) -> list[dict]:
        ds = self._ds()
        globbed = any(ch in pattern for ch in "*?[]")
        low = pattern.lower()
        base = ("MATCH (e:Entity {dataset:$ds}) WHERE NOT e:Parameter AND NOT e:File "
                "AND NOT e:Header AND NOT e:GlobalBlock AND ")
        ret = ("RETURN e.id AS id, e.kind AS kind, coalesce(e.qname,e.name) AS qname, "
               "e.name_lc AS name_lc, e.qname_lc AS qname_lc, e.file AS file, "
               "e.line_start AS line_start, coalesce(e.is_definition,true) AS is_definition ")
        with self._session() as s:
            if globbed:
                # translate a glob to a Cypher-safe regex
                rx = "^" + re.escape(low).replace(r"\*", ".*").replace(r"\?", ".") + "$"
                rows = s.run(
                    base + "(e.qname_lc =~ $rx OR e.name_lc =~ $rx) " + ret +
                    "ORDER BY size(coalesce(e.qname,e.name)), qname LIMIT $k",
                    rx=rx, k=limit * 4, ds=ds).data()
            else:
                rows = s.run(
                    base + "(e.qname_lc CONTAINS $low OR e.name_lc CONTAINS $low) " + ret +
                    "ORDER BY size(coalesce(e.qname,e.name)), qname LIMIT $k",
                    low=low, k=limit * 4, ds=ds).data()
        return rank_symbol_hits(rows, pattern, limit)

    def chunk(self, chunk_id: str) -> dict | None:
        ds = self._ds()
        with self._session() as s:
            rec = s.run(
                "MATCH (c:Chunk {dataset:$ds}) WHERE c.id = $id "
                "RETURN c.id AS id, c.kind AS kind, "
                "c.qname AS qname, c.file AS file, c.line_start AS line_start, "
                "c.line_end AS line_end, c.text AS text LIMIT 1", id=chunk_id,
                ds=ds).single()
            if rec:
                return dict(rec)
            ent = self.entity(chunk_id)
            if ent and ent.get("code"):
                return {"id": ent["id"], "kind": ent.get("kind"),
                        "qname": ent.get("qname"), "file": ent.get("file"),
                        "line_start": ent.get("line_start"),
                        "line_end": ent.get("line_end"), "text": ent["code"]}
        return None

    # ---- housekeeping ----------------------------------------------------
    def stats(self, dataset: str | None = None) -> dict:
        ds = dataset if dataset is not None else self.dataset
        with self._session() as s:
            if ds:
                n = s.run("MATCH (e:Entity {dataset:$ds}) RETURN count(e) AS n",
                          ds=ds).single()["n"]
                c = s.run("MATCH (c:Chunk {dataset:$ds}) RETURN count(c) AS n",
                          ds=ds).single()["n"]
                r = s.run("MATCH (a:Entity {dataset:$ds})-[x]->() "
                          "WHERE x.ords IS NOT NULL "
                          "RETURN sum(size(x.ords)) AS n", ds=ds).single()["n"]
                return {"dataset": ds, "entities": n, "chunks": c,
                        "relations": r or 0}
            n = s.run("MATCH (e:Entity) WHERE e.dataset IS NOT NULL "
                      "RETURN count(e) AS n").single()["n"]
            c = s.run("MATCH (c:Chunk) WHERE c.dataset IS NOT NULL "
                      "RETURN count(c) AS n").single()["n"]
            r = s.run("MATCH ()-[x]->() WHERE x.ords IS NOT NULL "
                      "RETURN sum(size(x.ords)) AS n").single()["n"]
            legacy = s.run("MATCH (e:Entity) WHERE e.dataset IS NULL "
                           "RETURN count(e) AS n").single()["n"]
        out = {"entities": n, "chunks": c, "relations": r or 0,
               "datasets": self.datasets()}
        if legacy:
            out["legacy_nodes_without_dataset"] = legacy
        return out

    def close(self) -> None:
        if self._owns_driver:
            self.driver.close()
def _fts_query(query: str) -> str:
    """Turn a natural-language question into a Lucene-friendly OR query.

    Stop-words carry no signal and a bare question often matches nothing, so
    terms are OR-ed: better a weak ranking than an empty result.
    """
    words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)]
    stop = {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "how",
            "what", "why", "when", "where", "which", "who", "to", "of", "in", "on",
            "for", "and", "or", "it", "this", "that", "with", "from", "by", "be",
            "if", "not", "i", "we", "you", "they", "there", "their", "can", "will"}
    keep, seen = [], set()
    for w in words:
        lw = w.lower()
        if lw in stop or lw in seen:
            continue
        seen.add(lw)
        # Lucene's standard analyzer does not stem, so "exceptions" would miss
        # "exception". A single edit of slack fixes plurals and inflections
        # without turning short terms into noise.
        keep.append(f"{w}~1" if len(w) >= 5 else w)
    return " OR ".join(keep[:16])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_store(onto: oc.CppOntology | None = None, kind: str | None = None,
               payload: dict | None = None) -> GraphStore:
    """Build the configured store.

    `kind` overrides PROD_RAG_STORE. The Neo4j backend must already hold the
    index (see `prod-rag index add --store neo4j`); it is not fed from `onto`.
    """
    kind = (kind or store_kind()).lower()
    if kind == "neo4j":
        if not payload or not payload.get("name"):
            raise ValueError("a Neo4j store needs payload['name'] to pick its dataset")
        return Neo4jStore(dataset=payload["name"])
    if onto is None:
        raise ValueError("MemoryStore needs an ontology")
    return MemoryStore(onto)
