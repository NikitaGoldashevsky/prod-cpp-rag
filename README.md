# PROD_RAG

Codebase knowledge for agents and people.

PROD_RAG parses C/C++ with tree-sitter, builds an **ontology** (classes, methods,
fields, calls, inheritance, exception sites) plus source **chunks**, stores it in
**Neo4j**, and serves that knowledge two ways:

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

## Storage: Neo4j

The graph lives in a local **Neo4j Community** instance. Ingestion, entity
lookup, neighbourhood expansion and lexical search are all Cypher — the Python
side parses source and renders fact cards, never scans the graph.

Why this is the default: on a 500-file / 185k-line tree the ontology is ~40k
entities and ~204k relations. Scoring every entity in Python took 416–848 ms per
query; Neo4j answers the same queries in **19–169 ms**. A memory backend still
exists (`--store memory`) for tests and machines without a database, and
`tests/test_store_parity.py` proves the two agree on content.

### Installing Neo4j (no root, no Docker)

`./neo4j_setup.sh` in the diploma repo does this; the steps are:

```bash
# one-shot: pull the official image layers over HTTPS, verify digests,
# unpack under ~/.local/opt/neo4j (ships its own JDK — no Java install)
./neo4j_setup.sh
~/.local/bin/neo4j-local console        # or: start / stop / status
```

Credentials land in `~/.local/opt/neo4j/credentials.env` (mode 600). PROD_RAG
reads `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD`, falling back to that file.

If you already run Neo4j (including Neo4j Desktop or Aura), just export:

```bash
export NEO4J_URI=neo4j://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=...
```

**Sizing:** the image rootfs is ~1.2 GB on disk (367 MB of layers). Heap is set
to 1 GB and pagecache to 512 MB in the shipped config; a 500-file index fits
comfortably. Verify with `prod-rag neo4j status`.

The Neo4j Browser at `http://127.0.0.1:7474` is useful for the thesis: run
`MATCH p=(a:Entity)-[:CALLS*1..3]->(b) RETURN p LIMIT 50` and show the graph.

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
export PATH="$HOME/prod-cpp-rag/.venv/bin:$PATH"
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
prod-rag index add core     ~/work/super/submodules/core --store neo4j
prod-rag index add network  ~/work/super/submodules/network --store neo4j
prod-rag index list
```

`--store neo4j` pushes the parsed graph into Neo4j as part of the build (and
skips the Python BM25 index, since Lucene does that work). Without the flag the
target comes from `PROD_RAG_STORE` (default `memory`).

Each index occupies its own **dataset** in the graph: every entity and chunk
carries a `dataset` property, and every query filters on it. So `core` and
`network` can be live at the same time, and a query for one can never be
answered out of the other.

If an index exists on disk but not in the graph, the tools **refuse** rather
than answer from whatever else is loaded:

```json
{"error": "store_not_ready",
 "detail": "index 'core' is not in the Neo4j graph, so answering would use the
            wrong data. Push it:  prod-rag neo4j push core\n graph holds: network"}
```
`list_indexes` shows the same thing up front as `store: {ready: false, hint: …}`.

To push an index you already built:

```bash
prod-rag neo4j push core          # replaces the 'core' dataset only
prod-rag neo4j push core --keep   # merge into the existing 'core' dataset
prod-rag neo4j push core --reset  # wipe EVERY dataset first (destructive)
prod-rag neo4j drop core          # remove one dataset from the graph
prod-rag neo4j status             # URI, counts, and the dataset table
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
  `third_party`, `external`, `dist`, `target`, `out`, `.venv`, plus compiler
  output (`obj`, `CMakeFiles`) and Qt-generated files (`moc_*`, `ui_*`, `qrc_*`).
  Generated `moc_` classes shadow the real ones and make a class name resolve to
  a body-less stub, so they are skipped even inside a directory you index.

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
get_entity(name, index="")           signature, callers, callees, throws, source,
                                     declared_in_index / first_member for classes
get_neighbours(name, index="", hops=1) one-hop graph expansion
find_symbol(pattern, index="", limit=50) substring or glob, ranked
read_chunk(chunk_id, index="")       one chunk by id
reindex(name="")                     rebuild after edits (index only, never source)
answer_question(question)            only if a model endpoint is configured
```

Every tool is read-only on the filesystem except `reindex`, which writes to the
index store under `~/.local/share/prod-rag`.

`find_symbol` returns every hit with a `match` field (`exact`, `prefix`, `word`,
`substring`, `glob`), ordered best first. Substring noise is real — the letters
`tcp` genuinely occur inside `adjustCpu` — so a hit that only matched mid-word
is labelled as such and never sorted above a real symbol. If every hit says
`substring`, the symbol you named probably does not exist.

### Wiring it into a harness

Any MCP client needs the command, its arguments, and optionally `PROD_RAG_DATA`.
The canonical entry is in `examples/mcp-config.json`:

```json
{
  "mcpServers": {
    "prod-rag": {
      "command": "/home/YOU/PROD_RAG/.venv/bin/prod-rag",
      "args": ["mcp", "--store", "neo4j"]
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
<harness> mcp add prod-rag /home/YOU/PROD_RAG/.venv/bin/prod-rag mcp --store neo4j
```

`--store neo4j` is not optional if the graph is the source of truth. With the
memory default the server answers from the JSON index instead — same answers for
structure, but it re-parses that index per process (seconds on a large
submodule) and the Neo4j dataset checks in `list_indexes` disappear.

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
prod-rag neo4j status               connection, counts and the dataset table
prod-rag neo4j push <name>          push/replace one dataset (--reset wipes all)
prod-rag neo4j drop <name>          remove one dataset
prod-rag search "who calls X"       facts + chunks in the terminal
prod-rag entity TcpClient::connect  full JSON for one entity
prod-rag serve --store neo4j        web UI
prod-rag mcp --store neo4j          MCP server
```

Select the backend with `--store` or `PROD_RAG_STORE=neo4j`. The MCP server and
the web UI accept the same flag, so one index serves both.

---

## How it works

```
C/C++ files
   │  tree-sitter parse
   ▼
entities + relations ── chunker ──► chunks (method / class / field, with file:line)
   │                                  │
   │ fact cards (exact [FACT] lines)  │
   ▼                                  ▼
        ┌─────────────────  Neo4j  ─────────────────┐
        │ (:Entity)-[:CALLS|HAS_METHOD|THROWS ...]  │
        │ (:Chunk)-[:DESCRIBES]->(:Entity)          │
        │ full-text: entity_text, chunk_text        │
        └───────────────┬───────────────────────────┘
                        │ Cypher
        query ──► full-text match ──► 1-hop expansion ──► fact cards + chunks
                                                        │
                                    ┌───────────────────┴──────────────────┐
                                    ▼                                      ▼
                            MCP tools (evidence)                  Q&A (model + verifier)
```

Graph shape:

- `(:Entity {id, kind, name, qname, file, line_start, line_end, fact_card, ...})`
  plus a kind label (`:Class`, `:Method`, `:Field`, …). `id` is a unique property.
- `(:Chunk {id, text, ...})-[:DESCRIBES]->(:Entity)`.
- Predicates are relationship types: `CALLS`, `HAS_METHOD`, `HAS_FIELD`,
  `INITIALIZES`, `INCLUDES`, `DECLARES`, `REFERENCES`, `HAS_PARAMETER`.
- Literal objects become stub nodes: `CALLS_EXTERNAL → (:ExternalSymbol)`,
  `INHERITS_FROM → (:ExternalBase)`, `THROWS {guarded, line} → (:ThrowType)`, so
  "which calls propagate exceptions unhandled?" is a Cypher predicate.
- Multiplicity is preserved: an edge carries `ords`, the list of source
  positions that produced it, so `std::to_string` called three times stays three
  facts.
- Two full-text indexes (`entity_text`, `chunk_text`) do the lexical retrieval
  that BM25 did in the memory backend.

Entities: `File, Header, Class, Struct, Method, Function, Constructor,
Destructor, Field, Parameter, TypeAlias, GlobalBlock`.
Relations: `includes, declares, hasMethod, hasConstructor, hasField,
hasParameter, inheritsFrom, calls, references, initializes, throws, aliasOf`,
plus control-flow facts per callable: `has_try_catch` and `throw_sites`
(each tagged `guarded` / `unguarded`).

Exception handling is modelled as a fact, not a text match — that is what makes
"is this call guarded?" answerable.

### The call graph, and how far to trust it

`CALLS` is resolved by C++ scoping, not by name matching:

1. the caller's own class, then its bases (`SingleTimer::toLable` calling
   `deltaTime()` finds `AlarmClock::deltaTime`);
2. a receiver with a known type — a parameter, a local (`QString res;`), or a
   field of the caller or a base (`client_.connect()` → `TcpClient::connect`);
3. a free function in the same file, then the same directory;
4. a free function that is unambiguous in the whole corpus;
5. `Type(args)` → that type's constructor.

Anything else becomes `CALLS_EXTERNAL` with the **receiver kept** (`res.append`,
`ip_.c_str`, `::connect`), which is the honest answer: the callee is outside the
indexed tree or its receiver type is unknown. An unknown receiver is never
guessed — that guess is what made a `QString::append` call look like
`TextForm::append`.

Two consequences worth knowing:

- A call through a **container or smart pointer** (`vector_.size()`) lands in
  `CALLS_EXTERNAL`: the receiver type is `std::vector`, not the element type.
- A leading `::` means global scope, so `::connect(...)` inside
  `TcpClient::connect` is the libc call and produces no self-edge.

Measured on a generated 200-caller corpus built to reproduce the reported
failures (same-named methods in every class, inheritance chains, member calls
through locals and fields, explicit `::` calls): precision **0.857 → 1.000**,
recall **0.750 → 1.000**. On `cct-app` the edges that changed were exactly the
bad ones: `Server::handleClient → Vector3::toString` (a name collision) is gone,
`config::loadFromJson → DeviceState` is now `DeviceState::DeviceState`, and
`Server::handleClient → Vector3::distanceTo` now appears.

Classes: `get_entity` reports `declared_in_index` and, when a class is known only
through its method definitions (its header lives outside the indexed subtree), a
`first_member` you can jump to instead of `line_start: 0`.

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
- **The call graph is a static, single-file-scope approximation.** It resolves
  through declared types, not through templates, `auto`, or virtual dispatch: a
  call through an unknown variable is reported as external rather than guessed,
  and overrides are not linked to the base declaration. Treat `calls`/`callers`
  as strong evidence and confirm with the source when the answer turns on it.
- **Generic words can drown specific ones.** In a large project, "client" may
  link a field named `client_` before the class that actually answers the
  question. Prefer naming the symbol.
- **Freshness is reported, not enforced.** A stale index still answers; call
  `index_status` or `reindex`. Reindexing updates the file index only — the
  graph then lags it until `prod-rag neo4j push <name>`; `list_indexes` reports
  that as a `store.hint`.
- **Indexes are whole-subtree.** There is no incremental per-file update yet;
  a rebuild re-parses and re-pushes the whole indexed subtree.
- **The Neo4j graph holds several datasets, but the index loader is one graph.**
  `neo4j push` replaces only its own dataset. The memory backend rebuilds a
  BM25 index per index on first use; the Neo4j backend skips that entirely.
- **Generated code is skipped, but old indexes keep it.** `moc_*`/`ui_*`/`qrc_*`
  files and `obj/`, `CMakeFiles/` trees are excluded from new builds. Indexes
  built before that rule need a rebuild to purge them.
- **Parsing, not querying, is the cost at scale.** 500 files / 185k lines →
  31 s to parse and push 40k entities; queries then run in 19–169 ms.

  Measured (this machine):

  | files | lines | entities | build+push | rescan | search (mem) | search (Neo4j) |
  |-------|-------|----------|-----------|--------|--------------|----------------|
  | 32    | 3k    | 420      | 0.3 s     | 40 ms  | 11 ms        | 15 ms          |
  | 500   | 185k  | 40,005   | 31 s      | 13 ms  | 416–848 ms   | 19–169 ms      |

  Start with one submodule; grow as needed.
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

**`store_not_ready` from every tool** — the index is on disk but not in the
graph. `prod-rag neo4j push <name>`, then restart the MCP server (the dataset
check is cached per process). `prod-rag neo4j status` lists what the graph holds.

**Slow `index status` on a huge tree** — it hashes every file. Use
`--no-rescan`, or index a smaller subtree.
