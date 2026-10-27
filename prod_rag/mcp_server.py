# -*- coding: utf-8 -*-
"""
mcp_server.py — the MCP server: give an existing agent knowledge of a codebase.

Design decision worth stating: these tools return **structured evidence**, not
generated prose. The agent already has a better model than anything we would
wrap; what it cannot do is know the exact fact that `Server::run` has no
`try/catch`, or that `TcpClient::connect` is the function that dials out. So we
hand it entities, fact cards and source chunks with `file:line` provenance, and
let it write the answer.

Every tool is read-only with respect to the codebase. `reindex` writes an index
(under ~/.local/share/prod-rag), never source code.

Transports
----------
    prod-rag-mcp                      # stdio (default; the harness spawns it)
    prod-rag-mcp --transport http     # streamable HTTP for a shared instance
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import index as idx_mod
from . import llm as llm_mod
from . import store as store_mod
from .service import Service

# ---------------------------------------------------------------------------
# Server construction
# ---------------------------------------------------------------------------


def _new_server(name: str):
    """Build an MCP server across SDK generations.

    mcp 1.x: `mcp.server.fastmcp.FastMCP`
    mcp 2.x: `mcp.server.mcpserver.MCPServer`  (FastMCP was renamed)
    Both expose `@server.tool()` and `server.run(transport=...)`.
    """
    try:
        from mcp.server.mcpserver import MCPServer          # mcp >= 2
        return MCPServer(name), "v2"
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP             # mcp 1.x
        return FastMCP(name), "v1"
    except ImportError as e:                               # pragma: no cover
        raise SystemExit(
            "the 'mcp' package is required for the MCP server:\n"
            "  pip install 'mcp'\n"
            f"(import error: {e})")


def build_server(service: Service, chat=None):
    """Create an MCP server exposing `service` as tools."""
    mcp, sdk = _new_server("prod-rag")

    def _dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=None)

    def _safe(fn, *a, **kw) -> str:
        """Run a service call, reporting a refusal as data the agent can act on.

        An unhandled exception reaches the client as "Error executing tool X"
        with the real reason only in the server's stderr — which the harness
        usually hides. The graph-missing and unknown-index cases are ordinary
        states, so they come back as a JSON body naming the fix.
        """
        try:
            return _dumps(fn(*a, **kw))
        except store_mod.StoreNotReady as e:
            return _dumps({"error": "store_not_ready", "detail": str(e)})
        except LookupError as e:
            return _dumps({"error": "lookup_failed", "detail": str(e)})

    @mcp.tool()
    def list_indexes() -> str:
        """List the codebase indexes this server can answer from.

        Each row carries its root path, size, build time, whether the working
        tree has changed since the index was built, and — on the Neo4j backend —
        whether that index is actually in the graph (`store.ready`). Call this
        first when you do not know which index covers the code you are looking at.
        """
        return _safe(service.status, deep=False)

    @mcp.tool()
    def index_status(name: str = "") -> str:
        """Report whether an index still matches the code on disk.

        Returns per-folder content hashes with a status of fresh / stale / new /
        gone, plus the list of modified, added and removed files. Call this
        before trusting answers about code that may have changed, and after you
        edit files.
        """
        def run():
            if name:
                idx = service.get(name)
                return {"name": idx.name, "root": idx.root,
                        **idx.freshness(rescan=True)}
            return service.status(deep=True)
        return _safe(run)

    @mcp.tool()
    def search_code(query: str, index: str = "", top_k: int = 8) -> str:
        """Find the code and ontology facts relevant to a natural-language query.

        Returns linked ontology entities (with file:line), the exact structured
        fact cards, and the best matching source chunks. This is the primary
        entry point for "where is X", "what does X do", "who calls X".

        Each linked entity carries `match`: `qname` or `name` means the query
        named it; `text` means it merely shares words with the query. Trust the
        first two, treat `text` hits as leads. If the symbol you asked about is
        absent entirely, it does not exist under that name in this index.
        """
        return _safe(service.search, query, index=index or None,
                     top_k=max(1, min(top_k, 30)))

    @mcp.tool()
    def get_entity(name: str, index: str = "") -> str:
        """Look up one entity by name or qualified name.

        Gives the signature, parameters, modifiers, fields it initialises,
        inheritance, every call it makes and every call made to it, the
        exception sites (with whether each is inside a try/catch), the exact
        fact card, and the entity's source text.

        `calls` is scope-resolved (own class, bases, typed receivers, same-file
        free functions). Callees it could not resolve are listed in `uses` with
        their receiver — treat those as "unknown or outside the index", never as
        a call to a project function. For a class, `declared_in_index: false`
        means its header is outside the indexed subtree, and `first_member`
        points at real code instead of an empty line range.

        If the name is overloaded the response carries `overloaded: true` and an
        `overloads` list (id, parameters, location) — the `calls`/`callers`
        fields then describe the one overload that was returned. Re-query with
        an id from that list to see another. Both a bare id and an
        `index::id` id (as printed by search_code) are accepted.
        """
        return _safe(service.entity, name, index=index or None)

    @mcp.tool()
    def get_neighbours(name: str, index: str = "", hops: int = 1) -> str:
        """Expand one entity one or two hops over the code graph.

        Use this when a question spans several symbols ("how does the client
        connection reach the logger?"). Returns the surrounding entities and
        their fact cards.
        """
        return _safe(service.neighbours, name, index=index or None,
                     hops=max(1, min(hops, 3)))

    @mcp.tool()
    def find_symbol(pattern: str, index: str = "", limit: int = 50) -> str:
        """Find entities by substring or glob over their names.

        `pattern` is case-insensitive; `*`, `?` and `[]` are glob wildcards.
        Example: `find_symbol("Tcp*")`, `find_symbol("handle")`.

        Every hit carries a `match` field: exact / prefix / word / substring /
        glob. Hits are ordered best-first, so a substring that happens to sit
        mid-word ("Tcp" inside "adjusT-Cpu") never outranks a real symbol.
        If you only see `substring` hits, the symbol you named probably does not
        exist — check those before concluding it does.
        """
        return _safe(service.find, pattern, index=index or None,
                     limit=max(1, min(limit, 200)))

    @mcp.tool()
    def read_chunk(chunk_id: str, index: str = "") -> str:
        """Read one source chunk by id (ids come from search_code).

        Accepts `index::chunk-id` or a bare chunk id. Returns the exact text
        with file, line range and kind.
        """
        return _safe(service.chunk, chunk_id, index=index or None)

    @mcp.tool()
    def reindex(name: str = "") -> str:
        """Rebuild an index after files changed.

        Writes only to the index store, never to the codebase. Call it after you
        add, edit or delete C/C++ files, then continue with search_code.
        """
        names = [name] if name else service.names()
        if not names:
            return _dumps({"error": "no indexes loaded"})
        out = []
        for n in names:
            meta = service.reindex(n)
            out.append(meta)
        return _dumps({"reindexed": out})

    if chat is not None:
        @mcp.tool()
        def answer_question(question: str, index: str = "", top_k: int = 5) -> str:
            """Answer a question with the full retrieve -> generate -> verify path.

            This runs a local model and is slower and less capable than you are.
            Prefer search_code / get_entity and write the answer yourself; use
            this only when a grounded, citation-carrying draft is wanted.
            """
            return _safe(service.answer, question, chat,
                         index=index or None, top_k=max(1, min(top_k, 20)))

    return mcp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="prod-rag-mcp",
        description="MCP server exposing codebase knowledge from PROD_RAG indexes")
    ap.add_argument("--transport", choices=("stdio", "http", "sse"), default="stdio",
                    help="stdio (default) or streamable HTTP for a shared instance")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--index", action="append", metavar="NAME",
                    help="load only these indexes (default: all built indexes)")
    ap.add_argument("--llm-base-url", default=os.environ.get("PROD_RAG_LLM_BASE_URL"),
                    help="enable answer_question() with an OpenAI-compatible endpoint")
    ap.add_argument("--llm-model", default=os.environ.get("PROD_RAG_LLM_MODEL"))
    ap.add_argument("--llm-api-key", default=os.environ.get("PROD_RAG_LLM_API_KEY"))
    ap.add_argument("--store", choices=("memory", "neo4j"),
                    default=os.environ.get("PROD_RAG_STORE"),
                    help="graph backend (default $PROD_RAG_STORE or memory)")
    ap.add_argument("--list-indexes", action="store_true",
                    help="print known indexes and exit (for install scripts)")
    args = ap.parse_args(argv)

    if args.list_indexes:
        for m in idx_mod.list_indexes():
            s = m.get("stats", {})
            print(f"{m['name']}\t{m['root']}\t{s.get('files', '?')} files\t"
                  f"{s.get('entities', '?')} entities\t{m.get('updated_at', '')}")
        return 0

    service = Service(args.index, store_kind=args.store)
    if not service.names():
        print("warning: no indexes built yet — the tools will report none.\n"
              "         build one with: prod-rag index add <name> <path>",
              file=sys.stderr)

    chat = None
    if args.llm_base_url:
        chat = llm_mod.OpenAIChat(args.llm_base_url, model=args.llm_model,
                                  api_key=args.llm_api_key)

    mcp = build_server(service, chat)

    if args.transport == "stdio":
        mcp.run()
        return 0

    transport = "streamable-http" if args.transport == "http" else "sse"
    # mcp 2.x takes host/port as run() kwargs; mcp 1.x reads them from settings.
    try:
        mcp.run(transport=transport, host=args.host, port=args.port)
    except TypeError:
        for attr, value in (("host", args.host), ("port", args.port)):
            try:
                setattr(mcp.settings, attr, value)
            except Exception:
                pass
        mcp.run(transport=transport)
    return 0


if __name__ == "__main__":
    sys.exit(main())
