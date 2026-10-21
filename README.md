# PROD_RAG

Codebase knowledge for agents and people.

PROD_RAG parses C/C++ with tree-sitter, builds an **ontology** (classes, methods,
fields, calls, inheritance, exception sites) plus source **chunks**, and serves
that knowledge two ways:

- **MCP server** — so an existing agent (Codewhale, Claude, Cursor, anything that
  speaks MCP) can ask what is *actually* in the codebase instead of guessing.
- **Web UI** — so a person can ask a question and see the answer *with its
  evidence*, browse the code graph, and check whether the index still matches
  the working tree.

It is read-only with respect to your source code. Nothing here edits a file.

---

## Why this exists

A language model reading files one by one is good at local questions and bad at
structural ones: *who calls this*, *is this exception handled*, *what does this
class own*. Those answers are exact facts in the AST, not something to infer from
text. PROD_RAG extracts them once, indexes them, and hands them over.

**Design decision:** the MCP tools return **structured evidence, not prose**.
Your agent's model is better than anything we would wrap, and it can make
follow-up calls. What it cannot do is invent the fact that `Server::run` has no
`try/catch`, or that `TcpClient::connect` is the function that dials out. So the
tools return entities, fact cards and source chunks with `file:line` provenance.

The web UI's Q&A box is the exception: it runs the full
`retrieve → generate → verify` pipeline with a local model, because sometimes a
person wants one grounded paragraph rather than a browsing session.

---

## Install

Ubuntu 22.04 / any Linux with Python 3.10+:

```bash
git clone <this repo> ~/PROD_RAG
cd ~/PROD_RAG
./install.sh
```

That creates `.venv/`, installs `tree-sitter`, `tree-sitter-cpp` and `mcp`, and
puts three commands in `.venv/bin`:

```
prod-rag        the CLI (index, search, serve, mcp, doctor)
prod-rag-mcp    the MCP server  (same as `prod-rag mcp`)
prod-rag-web    the web UI      (same as `prod-rag serve`)
```

Add it to your PATH:

```bash
export PATH="$HOME/PROD_RAG/.venv/bin:$PATH"
```

Verify:

```bash
prod-rag doctor
```

---

## Index a codebase

An **index** is a named snapshot of one directory subtree. You do not index the
whole monorepo — you index the submodule you are working in.

```bash
prod-rag index add core     ~/work/super/submodules/core
prod-rag index add network  ~/work/super/submodules/network
prod-rag index list
```

Starting small is the point: build one submodule, use it, then add more.

```bash
prod-rag index status              # freshness for every index
prod-rag index status core         # one index, with per-folder hashes
prod-rag index remove core         # forget it (the source is untouched)
```

Options:

- `--label NAME` — display name (defaults to the folder name).
- `--ext .cpp --ext .h` — restrict extensions (default: C/C++).
- `--exclude build --exclude third_party` — extra directories to skip.
  Defaults already skip `.git`, `build`, `cmake-build-*`, `node_modules`,
  `third_party`, `external`, `dist`, `target`, `out`, `.venv`.

Indexes live in `~/.local/share/prod-rag/indexes/<name>/index.json`
(override with `PROD_RAG_DATA`). They are not in this repository: they describe
*other* repositories. JSON, so a colleague can copy one or inspect it with `jq`.

### Freshness, and why folders have hashes

Every folder in an index carries two content hashes: one for the files directly
inside it, and a **subtree** hash that folds in every child folder. A change
three levels down therefore marks every ancestor stale — including the root.

```bash
$ prod-rag index status core
core  (/home/you/work/super/submodules/core)
  built   2026-10-21T10:05:19+0300
  state   stale — 1 modified, 0 added, 1 removed
FOLDER             FILES  SUBTREE  HASH                                  STATUS
-----------------  -----  -------  ------------------------------------  ------
.                  1      32       43cbe9b2399e4f87 -> 625129df0b95d73a  stale
commands           13     13       982eda8d179817f6                      fresh
network            4      4        3c85118f9c42e2c8 -> 30c390117083fc90  stale
tests              1      1        c55100b4624d0ea1 ->                   gone
  changed (1):
    network/TcpClient.cpp
  removed (1):
    tests/test_network.cpp
```

Exit codes: `0` fresh, `2` stale, `1` the indexed root is gone. That makes it
usable in a shell guard or a git hook.

An agent should call `index_status` before trusting an answer about code it may
have edited, and `reindex` afterwards.

---

## MCP server

```bash
prod-rag mcp                 # stdio — the harness spawns it (recommended)
prod-rag mcp --transport http --host 0.0.0.0 --port 8765    # one shared instance
```

`stdio` is the default and the safest: the harness starts the process, talks
over its stdin/stdout, and nothing is exposed on the network.

### Tools

```
list_indexes()                       indexes, roots, sizes, freshness summary
index_status(name="")                per-folder hashes + changed/added/removed
search_code(query, index="", top_k=8)  linked entities + fact cards + chunks
get_entity(name, index="")           signature, callers, callees, throws, source
get_neighbours(name, index="", hops=1) one-hop graph expansion
find_symbol(pattern, index="", limit=50) substring or glob over names
read_chunk(chunk_id, index="")       one chunk by id
reindex(name="")                     rebuild after edits (index only, never source)
answer_question(question)            only if a model endpoint is configured
```

Every tool is read-only on the filesystem except `reindex`, which writes to the
index store under `~/.local/share/prod-rag`.

### Wiring it into a harness

Any MCP client needs the command, its arguments, and optionally `PROD_RAG_DATA`.
The canonical entry is in `examples/mcp-config.json`:

```json
{
  "mcpServers": {
    "prod-rag": {
      "command": "/home/YOU/PROD_RAG/.venv/bin/prod-rag",
      "args": ["mcp"]
    }
  }
}
```

For a shared HTTP instance see `examples/mcp-config-http.json`
(`{"type": "http", "url": "http://host:8765/mcp"}`).

Where that JSON goes depends on the harness — Claude Desktop uses
`claude_desktop_config.json`, Cursor and Codewhale and most others accept a
project-level `.mcp.json` or an equivalent settings entry. If the harness has a
CLI for adding servers, use it and point it at the same command:

```bash
<harness> mcp add prod-rag /home/YOU/PROD_RAG/.venv/bin/prod-rag mcp
```

`AGENTS.md` in this repository is written to be handed to an agent so it can do
the installation itself.

### Sharing one instance

```bash
./scripts/serve-shared.sh      # MCP :8765 + web UI :8088, all interfaces
```

One process keeps one parsed index in memory and the whole team uses it; the
alternative is every laptop building its own. The MCP HTTP transport has no
built-in auth — bind it to a trusted network and firewall the port.

---

## Web UI

```bash
prod-rag serve --port 8088
# or with a model for the Q&A box:
prod-rag serve --port 8088 --llm-base-url http://192.168.10.46:8081
```

Open `http://127.0.0.1:8088`. Three views:

- **Ask** — the full `retrieve → generate → verify → (correct)` path. Shows the
  answer, the verifier's verdict (`supported` / `corrected` / `unsupported`),
  the exact fact cards the model received, and the chunks it read. Click a
  linked entity to jump to its details.
- **Browse** — search the graph without a model: find symbols, inspect an
  entity's signature, callers, callees, inheritance and exception sites.
- **Indexes** — every index with its build time and a per-folder hash table
  showing `fresh` / `stale` / `new` / `gone`, plus the changed-file list and a
  one-click rebuild.

Sharing it on the LAN:

```bash
prod-rag serve --host 0.0.0.0 --port 8088 --token "$(openssl rand -hex 16)"
# colleagues open http://<host>:8088/?token=<same token>
```

Without `--token`, binding to `0.0.0.0` exposes the index to anyone on the
network — the server prints a warning when you do that.

The Q&A box needs a model; everything else works without one. Any
OpenAI-compatible endpoint is accepted:

```bash
export PROD_RAG_LLM_BASE_URL=http://192.168.10.46:8081     # llama.cpp, vLLM, ...
export PROD_RAG_LLM_MODEL=...            # optional; auto-detected from /v1/models
export PROD_RAG_LLM_API_KEY=...          # only for hosted providers
```

---

## CLI reference

```bash
prod-rag doctor                     check python, deps, indexes, freshness
prod-rag index add <name> <path>    build an index
prod-rag index list                 names, roots, sizes, build times
prod-rag index status [name]        folder hashes (+ --no-rescan for speed)
prod-rag index remove <name>        forget an index
prod-rag search "who calls X"       facts + chunks in the terminal
prod-rag entity TcpClient::connect  full JSON for one entity
prod-rag serve                      web UI
prod-rag mcp                        MCP server
```

---

## How it works

```
C/C++ files
   │  tree-sitter parse
   ▼
entities + relations ── chunker ──► chunks (method / class / field, with file:line)
   │                                  │
   │ fact cards (exact [FACT] lines)  │ BM25 over chunk text + fact card
   ▼                                  ▼
query ──► entity linking ──► 1-hop neighbourhood ──► fact cards + ranked chunks
                                                        │
                                    ┌───────────────────┴──────────────────┐
                                    ▼                                      ▼
                            MCP tools (evidence)                  Q&A (model + verifier)
```

Entities: `File, Header, Class, Struct, Method, Function, Constructor,
Destructor, Field, Parameter, TypeAlias, GlobalBlock`.
Relations: `includes, declares, hasMethod, hasConstructor, hasField,
hasParameter, inheritsFrom, calls, references, initializes, throws, aliasOf`,
plus control-flow facts per callable: `has_try_catch` and `throw_sites`
(each tagged `guarded` / `unguarded`).

Exception handling is modelled as a fact, not a text match — that is what makes
"is this call guarded?" answerable.

### Repository layout

```
prod_rag/
  core.py         tree-sitter → ontology → chunks → BM25 → retrieval
  index.py        indexing, per-folder hashes, freshness, loaded index
  service.py      the facade both front ends call
  mcp_server.py   MCP tools (stdio / streamable HTTP)
  web.py          web UI (stdlib http.server, one HTML page)
  llm.py          optional OpenAI-compatible client for the Q&A box
  cli.py          the `prod-rag` command
tests/test_smoke.py   no network, no model; covers build/search/freshness/CLI
```

`core.py` is vendored from the diploma prototype (`ontology_rag/ontology_core.py`)
so this package is self-contained. Retrieval behaviour is thesis evidence — if
you change it, change it there too and re-run the parity check.

---

## Limits

- **C/C++ only.** Other languages parse to nothing.
- **One-hop entity linking.** A question that names nothing ("where are
  exceptions thrown unguarded?") links no entity; the search still returns the
  most relevant chunks and their fact cards, but it is weaker than an identifier
  query.
- **Generic words can drown specific ones.** In a large project, "client" may
  link a field named `client_` before the class that actually answers the
  question. Prefer naming the symbol.
- **Freshness is reported, not enforced.** A stale index still answers; call
  `index_status` or `reindex`.
- **Indexes are whole-subtree.** There is no incremental per-file update yet;
  a rebuild re-parses the subtree (fast: ~100 ms for 32 files, ~1 s for a few
  hundred).
- **Verification is same-model.** The Q&A verifier judges its own generator's
  draft. Useful, not independent.

---

## Troubleshooting

**`No module named mcp`** — `./install.sh` installs it; if you installed with
`pip install -e .`, add the extra: `pip install -e '.[mcp]'`.

**`no indexes yet`** — `prod-rag index add <name> <path>`, then restart the MCP
server (it loads indexes at startup).

**MCP server starts but the agent sees no tools** — check the harness is passing
the absolute path to `.venv/bin/prod-rag`; run `prod-rag mcp` by hand and
confirm it waits on stdin without printing to stdout (anything on stdout breaks
the protocol).

**Everything is stale right after building** — the tree changed between `index
add` and `index status`, or the index root points at a parent directory that
includes generated files. Rebuild, and add `--exclude`.

**Slow `index status` on a huge tree** — it hashes every file. Use
`--no-rescan`, or index a smaller subtree.
