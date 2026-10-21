# -*- coding: utf-8 -*-
"""PROD_RAG — codebase knowledge for agents and people.

Layers:
    core     tree-sitter C/C++ -> ontology (entities, relations, chunks)
    index    named indexes on disk, with per-folder content hashes
    service  one facade shared by the MCP server and the web UI
    mcp      read-only MCP tools (stdio / streamable HTTP)
    web      human UI: Q&A with evidence, browse, index freshness
"""
__version__ = "1.0.0"
