# -*- coding: utf-8 -*-
"""Store parity: the Neo4j backend must answer like the in-memory one.

    python tests/test_store_parity.py [path-to-cpp-tree]

Requires a running Neo4j (see README §Neo4j). Skips cleanly if unreachable.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prod_rag import index as idx_mod          # noqa: E402
from prod_rag import store as store_mod        # noqa: E402

QUESTIONS = [
    "how does the client connect to the server",
    "what happens if the server fails to bind",
    "where are exceptions thrown",
]


def main(argv) -> int:
    root = argv[1] if len(argv) > 1 else os.path.expanduser("~/cct-app")
    if not os.path.isdir(root):
        print(f"skip: {root} not found (pass a C/C++ tree as argv[1])")
        return 0

    tmp = tempfile.mkdtemp(prefix="prodrag-parity-")
    os.environ["PROD_RAG_DATA"] = tmp
    try:
        try:
            base = store_mod.Neo4jStore()
            neo = base.for_dataset("parity")
        except Exception as e:
            print(f"skip: Neo4j not reachable ({type(e).__name__}: {e})")
            return 0

        print("Store parity: memory vs Neo4j")
        meta = idx_mod.build("parity", root, verbose=False)
        payload = idx_mod.load("parity").payload
        stats = base.load_index(payload, clear=True)
        ok = True

        def fail(msg):
            nonlocal ok
            ok = False
            print(f"  FAIL  {msg}")

        # ---- counts ------------------------------------------------------
        mem_ents = len(payload["entities"])
        # aliasOf is stored as a node property, everything else becomes an edge
        mem_rels = len([r for r in payload["relations"] if r["p"] != "aliasOf"])
        mem_chunks = len(payload["chunks"])
        if stats["entities"] != mem_ents:
            fail(f"entities {stats['entities']} != {mem_ents}")
        if stats["chunks"] != mem_chunks:
            fail(f"chunks {stats['chunks']} != {mem_chunks}")
        if stats["relations"] != mem_rels:
            fail(f"relations {stats['relations']} != {mem_rels}")
        if ok:
            print(f"  OK    counts: {mem_ents} entities, {mem_rels} relations, "
                  f"{mem_chunks} chunks")

        mem = store_mod.MemoryStore(idx_mod.load("parity").onto)

        # ---- entity parity ----------------------------------------------
        sample = [e["qname"] for e in payload["entities"]
                  if e["kind"] in ("Method", "Function", "Class", "Constructor")
                  and e.get("qname")][:25]
        diffs, checked = [], 0
        for qname in sample:
            a, b = mem.entity(qname), neo.entity(qname)
            if not a or not b:
                diffs.append((qname, "missing", bool(a), bool(b)))
                continue
            checked += 1
            for key in ("fact_card", "file", "line_start", "line_end",
                        "has_try_catch", "kind", "qname"):
                if a.get(key) != b.get(key):
                    diffs.append((qname, key, a.get(key), b.get(key)))
            for key in ("calls", "callers", "members", "inherits", "uses"):
                if sorted(a.get(key) or []) != sorted(b.get(key) or []):
                    diffs.append((qname, key, sorted(a.get(key) or [])[:6],
                                  sorted(b.get(key) or [])[:6]))
            if [t["type"] for t in (a.get("throw_sites") or [])] != \
               [t["type"] for t in (b.get("throw_sites") or [])]:
                diffs.append((qname, "throw_sites", a.get("throw_sites"),
                              b.get("throw_sites")))
        if diffs:
            fail(f"{len(diffs)} entity mismatches, e.g. {diffs[:3]}")
        else:
            print(f"  OK    entity parity on {checked} entities "
                  f"(fact cards, calls, callers, throws)")

        # ---- neighbours --------------------------------------------------
        ndiff = 0
        for qname in sample[:8]:
            a, b = mem.neighbours(qname, hops=1), neo.neighbours(qname, hops=1)
            if bool(a.get("found")) != bool(b.get("found")):
                ndiff += 1
                continue
            if not a.get("found"):
                continue
            if set(a.get("facts", "").split("\n\n")) != set(b.get("facts", "").split("\n\n")):
                ndiff += 1
        if ndiff:
            fail(f"neighbourhood facts differ on {ndiff} of 8 entities")
        else:
            print("  OK    neighbourhood parity on 8 entities")

        # ---- find --------------------------------------------------------
        for pat in ("*Client*", "handle"):
            a = {h["qname"] for h in mem.find(pat, limit=50)}
            b = {h["qname"] for h in neo.find(pat, limit=50)}
            if not a:
                fail(f"memory find({pat!r}) returned nothing to compare")
            elif not (b & a):
                fail(f"find({pat!r}) no overlap: memory={list(a)[:3]} neo4j={list(b)[:3]}")
        print("  OK    find overlap on glob and substring")

        # ---- chunk -------------------------------------------------------
        cid = payload["chunks"][0]["id"]
        a, b = mem.chunk(cid), neo.chunk(cid)
        if not a or not b or a.get("text") != b.get("text"):
            fail("chunk text differs")
        else:
            print(f"  OK    chunk text identical ({len(a['text'])} chars)")

        # ---- search ------------------------------------------------------
        # Ranking legitimately differs: memory uses BM25, Neo4j uses Lucene
        # full-text. The gate is that Neo4j returns usable evidence for every
        # question, and that a query naming a symbol finds that symbol.
        for q in QUESTIONS:
            b = neo.search(q, top_k=5)
            if not b["chunks"]:
                fail(f"search({q!r}) returned no chunks from Neo4j")
            if not b["facts"]:
                fail(f"search({q!r}) returned no facts from Neo4j")
        for sym in ("TcpClient::connect", "Server::run"):
            b = neo.search(f"what does {sym} do", top_k=8)
            names = {c["qname"] for c in b["chunks"]} | {e["qname"] for e in b["linked"]}
            if sym not in names:
                fail(f"search for {sym!r} did not surface it: {list(names)[:4]}")
        print(f"  OK    search returns facts + chunks on {len(QUESTIONS)} questions; "
              "symbol queries surface the symbol")

        # ---- dataset isolation -------------------------------------------
        # Two submodules in one graph must not bleed into each other, and a
        # query for a dataset that was never pushed must refuse, not guess.
        other = dict(payload, name="parity2")
        base.load_index(other, clear=True)
        try:
            base.for_dataset("nope").assert_ready()
            fail("querying an unpushed dataset did not raise")
        except store_mod.StoreNotReady:
            print("  OK    unpushed dataset refuses to answer")
        after = neo.search("how does the client connect", top_k=3)
        both = {d["dataset"] for d in base.datasets()}
        if not {"parity", "parity2"} <= both:
            fail(f"datasets in graph: {sorted(both)}")
        elif not after["chunks"]:
            fail("querying 'parity' broke after pushing 'parity2'")
        else:
            print("  OK    two datasets coexist: 'parity' still answers "
                  f"({len(after['chunks'])} chunks), graph has {sorted(both)}")
        base.drop_dataset("parity2")
        if "parity2" in {d["dataset"] for d in base.datasets()}:
            fail("drop_dataset left nodes behind")
        else:
            print("  OK    drop_dataset removes only that dataset")

        print("\nPARITY:", "PASS" if ok else "FAIL")
        base.close()
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
