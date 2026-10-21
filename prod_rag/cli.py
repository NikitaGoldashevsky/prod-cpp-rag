# -*- coding: utf-8 -*-
"""cli.py — the `prod-rag` command."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import index as idx_mod
from . import store as store_mod


def _print_table(rows, headers):
    widths = [len(h) for h in headers]
    cells = []
    for r in rows:
        line = [str(c) for c in r]
        cells.append(line)
        for i, c in enumerate(line):
            widths[i] = max(widths[i], len(c))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for line in cells:
        print("  ".join(line[i].ljust(widths[i]) for i in range(len(headers))))


# ---------------------------------------------------------------------------
# index subcommands
# ---------------------------------------------------------------------------

def cmd_index_add(args) -> int:
    exts = tuple(args.ext) if args.ext else idx_mod.CPP_EXTS
    exclude = tuple(args.exclude) if args.exclude else idx_mod.DEFAULT_EXCLUDE
    target = (args.store or store_mod.store_kind()).lower()
    meta = idx_mod.build(args.name, args.path, exts=exts, exclude=exclude,
                         label=args.label, with_retriever=(target != "neo4j"))
    print(f"stored at {idx_mod.index_dir(args.name)}/index.json")

    if target == "neo4j":
        payload = idx_mod.load(args.name).payload
        st = store_mod.Neo4jStore()
        try:
            stats = st.load_index(payload, clear=not args.keep)
            print(f"pushed to Neo4j: {stats['entities']} entities, "
                  f"{stats['relations']} relations, {stats['chunks']} chunks")
        finally:
            st.close()
    return 0


def cmd_index_list(args) -> int:
    metas = idx_mod.list_indexes()
    if not metas:
        print("no indexes yet.  add one:  prod-rag index add <name> <path>")
        return 0
    rows = []
    for m in metas:
        s = m.get("stats", {})
        rows.append([m["name"], m["root"], s.get("files", "?"),
                     s.get("entities", "?"), s.get("chunks", "?"),
                     m.get("updated_at", "")])
    _print_table(rows, ["NAME", "ROOT", "FILES", "ENTITIES", "CHUNKS", "BUILT"])
    return 0


def cmd_index_status(args) -> int:
    names = [args.name] if args.name else [m["name"] for m in idx_mod.list_indexes()]
    if not names:
        print("no indexes")
        return 0
    rc = 0
    for name in names:
        idx = idx_mod.load(name)
        f = idx.freshness(rescan=not args.no_rescan)
        print(f"\n{name}  ({idx.root})")
        print(f"  built   {idx.built_at}")
        print(f"  state   {f['state']} — {f['detail']}")
        if f["state"] == "missing":
            rc = 1
            continue
        if f["state"] == "stale":
            rc = 2
        rows = [[d["path"], d["files"], d["subtree_files"],
                 d["hash"] + ("" if d["hash"] == d["hash_now"] else " -> " + d["hash_now"]),
                 d["status"]] for d in f["folders"]]
        _print_table(rows, ["FOLDER", "FILES", "SUBTREE", "HASH", "STATUS"])
        for label in ("changed", "added", "removed"):
            items = f.get(label) or []
            if items:
                print(f"  {label} ({len(items)}):")
                for it in items[:20]:
                    print(f"    {it}")
                if len(items) > 20:
                    print(f"    … {len(items) - 20} more")
    return rc


def cmd_neo4j_status(args) -> int:
    from . import store as store_mod
    try:
        st = store_mod.Neo4jStore()
    except Exception as e:
        print(f"Neo4j not reachable: {e}", file=sys.stderr)
        return 1
    try:
        st.ensure_schema()
        print(f"uri     {st.uri}")
        print("counts  " + json.dumps(st.stats()))
    finally:
        st.close()
    return 0


def cmd_neo4j_push(args) -> int:
    from . import store as store_mod
    payload = idx_mod.load(args.name).payload
    st = store_mod.Neo4jStore()
    try:
        stats = st.load_index(payload, clear=not args.keep)
        print("pushed: " + json.dumps(stats))
    finally:
        st.close()
    return 0


def cmd_index_remove(args) -> int:
    if idx_mod.remove(args.name):
        print(f"removed index {args.name!r}")
        return 0
    print(f"no index named {args.name!r}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# query commands
# ---------------------------------------------------------------------------

def _service(args):
    from .service import Service
    return Service(getattr(args, "index", None),
                   store_kind=getattr(args, "store", None))


def cmd_search(args) -> int:
    svc = _service(args)
    r = svc.search(args.query, index=None if args.index and len(args.index) > 1 else
                   (args.index[0] if args.index else None), top_k=args.top_k)
    groups = r.get("indexes") or [r]
    for g in groups:
        print(f"\n=== {g.get('index','')}  ({g.get('root','')})")
        if g.get("linked"):
            print("linked:", ", ".join(e["qname"] for e in g["linked"]))
        if g.get("facts"):
            print(g["facts"])
        for c in g.get("chunks", []):
            print(f"\n--- {c.get('kind')} {c.get('qname')} "
                  f"[{c.get('file')}:{c.get('line_start')}-{c.get('line_end')}]")
            print(c.get("text", ""))
    return 0


def cmd_entity(args) -> int:
    svc = _service(args)
    r = svc.entity(args.name, index=args.index[0] if args.index else None)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0 if r.get("found") else 1


def cmd_doctor(args) -> int:
    ok = True
    print("PROD_RAG doctor")
    print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
    try:
        import tree_sitter, tree_sitter_cpp
        print(f"  tree_sitter {tree_sitter.__version__ if hasattr(tree_sitter,'__version__') else '?'}")
    except Exception as e:
        ok = False
        print(f"  tree_sitter MISSING: {e}  ->  pip install -r requirements.txt")
    try:
        import mcp
        print(f"  mcp         {getattr(mcp, '__version__', 'installed')}")
    except Exception:
        print("  mcp         not installed (MCP server disabled; pip install 'mcp>=1.2')")
    kind = store_mod.store_kind()
    print(f"  store       {kind}")
    if kind == "neo4j":
        try:
            neo = store_mod.Neo4jStore()
            print(f"    neo4j     {neo.uri} {json.dumps(neo.stats())}")
            neo.close()
        except Exception as e:
            ok = False
            print(f"    neo4j     UNREACHABLE: {e}")
    metas = idx_mod.list_indexes()
    print(f"  data dir    {idx_mod.data_dir()}")
    print(f"  indexes     {len(metas)}")
    for m in metas:
        idx = idx_mod.load(m["name"])
        f = idx.freshness(rescan=True)
        print(f"    - {m['name']:<18} {f['state']:<8} {m['root']}")
    if not metas:
        ok = False
        print("    (none — build one: prod-rag index add <name> <path>)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="prod-rag",
        description="Codebase knowledge for agents and people: indexes, MCP tools, web UI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # index ...
    idx = sub.add_parser("index", help="build and inspect indexes")
    isub = idx.add_subparsers(dest="sub", required=True)

    a = isub.add_parser("add", help="parse a directory subtree into a named index")
    a.add_argument("name")
    a.add_argument("path")
    a.add_argument("--label", help="human-friendly label (default: folder name)")
    a.add_argument("--ext", action="append",
                   help="file extension to include (repeatable; default: C/C++)")
    a.add_argument("--exclude", action="append",
                   help="directory name to skip (repeatable)")
    a.add_argument("--store", choices=("memory", "neo4j"),
                   help="also push this index into a backend (default: $PROD_RAG_STORE)")
    a.add_argument("--keep", action="store_true",
                   help="with --store neo4j: keep existing graph contents")
    a.set_defaults(func=cmd_index_add)

    l = isub.add_parser("list", help="list indexes")
    l.set_defaults(func=cmd_index_list)

    s = isub.add_parser("status", help="show folder hashes and freshness")
    s.add_argument("name", nargs="?")
    s.add_argument("--no-rescan", action="store_true",
                   help="do not hash the working tree (fast, always reports last state)")
    s.set_defaults(func=cmd_index_status)

    r = isub.add_parser("remove", help="forget an index")
    r.add_argument("name")
    r.set_defaults(func=cmd_index_remove)

    # search / entity
    se = sub.add_parser("search", help="retrieve facts and chunks for a query")
    se.add_argument("query")
    se.add_argument("--index", action="append")
    se.add_argument("--top-k", type=int, default=6)
    se.add_argument("--store", choices=("memory", "neo4j"),
                    default=os.environ.get("PROD_RAG_STORE"))
    se.set_defaults(func=cmd_search)

    en = sub.add_parser("entity", help="look up one entity")
    en.add_argument("name")
    en.add_argument("--index", action="append")
    en.add_argument("--store", choices=("memory", "neo4j"),
                    default=os.environ.get("PROD_RAG_STORE"))
    en.set_defaults(func=cmd_entity)

    # neo4j
    nj = sub.add_parser("neo4j", help="manage the Neo4j backend")
    jsub = nj.add_subparsers(dest="jsub", required=True)
    js = jsub.add_parser("status", help="connection + graph counts")
    js.set_defaults(func=cmd_neo4j_status)
    jp = jsub.add_parser("push", help="push an existing index into Neo4j")
    jp.add_argument("name")
    jp.add_argument("--keep", action="store_true", help="do not wipe existing data")
    jp.set_defaults(func=cmd_neo4j_push)

    # doctor
    d = sub.add_parser("doctor", help="check the installation")
    d.set_defaults(func=cmd_doctor)

    # serve
    w = sub.add_parser("serve", help="start the web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8088)
    w.add_argument("--index", action="append")
    w.add_argument("--llm-base-url", default=os.environ.get("PROD_RAG_LLM_BASE_URL"))
    w.add_argument("--llm-model", default=os.environ.get("PROD_RAG_LLM_MODEL"))
    w.add_argument("--llm-api-key", default=os.environ.get("PROD_RAG_LLM_API_KEY"))
    w.add_argument("--token", default=os.environ.get("PROD_RAG_TOKEN"))
    w.add_argument("--store", choices=("memory", "neo4j"),
                   default=os.environ.get("PROD_RAG_STORE"))
    w.add_argument("--top-k", type=int, default=5)
    w.add_argument("--max-chars", type=int, default=6000)
    w.set_defaults(func=_cmd_serve)

    # mcp
    m = sub.add_parser("mcp", help="start the MCP server (stdio by default)")
    m.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8765)
    m.add_argument("--index", action="append")
    m.add_argument("--llm-base-url", default=os.environ.get("PROD_RAG_LLM_BASE_URL"))
    m.add_argument("--llm-model", default=os.environ.get("PROD_RAG_LLM_MODEL"))
    m.add_argument("--llm-api-key", default=os.environ.get("PROD_RAG_LLM_API_KEY"))
    m.add_argument("--store", choices=("memory", "neo4j"),
                   default=os.environ.get("PROD_RAG_STORE"))
    m.set_defaults(func=_cmd_mcp)
    return ap


def _cmd_serve(args) -> int:
    from . import web
    argv = ["--host", args.host, "--port", str(args.port),
            "--top-k", str(args.top_k), "--max-chars", str(args.max_chars)]
    for name in (args.index or []):
        argv += ["--index", name]
    for flag, value in (("--llm-base-url", args.llm_base_url),
                        ("--llm-model", args.llm_model),
                        ("--llm-api-key", args.llm_api_key),
                        ("--token", args.token),
                        ("--store", args.store)):
        if value:
            argv += [flag, value]
    return web.main(argv)


def _cmd_mcp(args) -> int:
    from . import mcp_server
    argv = ["--transport", args.transport, "--host", args.host, "--port", str(args.port)]
    for name in (args.index or []):
        argv += ["--index", name]
    for flag, value in (("--llm-base-url", args.llm_base_url),
                        ("--llm-model", args.llm_model),
                        ("--llm-api-key", args.llm_api_key),
                        ("--store", args.store)):
        if value:
            argv += [flag, value]
    return mcp_server.main(argv)


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, LookupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
