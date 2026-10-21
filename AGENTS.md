# AGENTS.md — install and configure PROD_RAG on this machine

You are reading this because a human asked you to set up **PROD_RAG**: a
read-only codebase-knowledge service (MCP server + web UI) on this machine.

Follow the steps in order. Each one has a **verify** line — do not continue past
a step whose verification fails. Report what you did, what you verified, and
anything that did not work.

If a command needs `sudo` and you cannot get it, stop and report exactly which
package was missing; do not attempt workarounds that modify system Python.

---

## 0. Facts you need before starting

| | |
|---|---|
| Repo location | ask the human, or use `~/PROD_RAG` |
| Python needed | 3.10+ (Ubuntu 22.04 ships 3.10 — fine) |
| Dependencies | `tree-sitter`, `tree-sitter-cpp`, `mcp`, `neo4j` — all wheels, no compiler |
| Neo4j | **required**; see step 2 |
| Source code touched | **none** — PROD_RAG is read-only on the codebase |
| State written | `~/.local/share/prod-rag/` (indexes only) |
| Network | none required at runtime; the model is optional |

---

## 1. Install

```bash
cd <repo>                 # e.g. ~/PROD_RAG
./install.sh
```

`install.sh` creates `./.venv`, installs dependencies, and installs the package
in editable mode. It needs no root.

**Verify:**

```bash
./.venv/bin/prod-rag doctor
```

Expected: it prints the Python version, `tree_sitter` and `mcp` versions, the
active store, the data dir, and `indexes 0`. It exits non-zero only when no index
exists yet — that is expected at this point.

If `python3 -m venv` fails: `sudo apt-get install -y python3-venv`, then rerun.

---

## 2. Ensure Neo4j is running

The graph lives in Neo4j. Check first, install only if missing.

```bash
cd <repo> && ./.venv/bin/prod-rag neo4j status
```

- Prints `uri` and `counts` → it is running, go to step 3.
- `Neo4j not reachable` → install it:

  ```bash
  bash ./neo4j_setup.sh                  # no root, no Docker; ~1.2 GB, ships a JDK
  setsid nohup ~/.local/bin/neo4j-local console > /tmp/neo4j-console.log 2>&1 &
  sleep 20 && ./.venv/bin/prod-rag neo4j status
  ```

  Credentials are generated into `~/.local/opt/neo4j/credentials.env` (mode 600)
  and read automatically. **Verify:** `neo4j status` prints a `uri` line and
  non-zero counts, or zeros if the graph is empty.

If the machine already has Neo4j, export `NEO4J_URI`, `NEO4J_USER`,
`NEO4J_PASSWORD` instead of installing anything.

## 3. Build the first index

Ask the human which submodule to start with. **Index one submodule, not the whole
monorepo** — the full tree is too large and the first index should be quick to
validate.

```bash
./.venv/bin/prod-rag index add <short-name> <absolute-path-to-submodule> --store neo4j
```

Naming: short, lowercase, no spaces — `core`, `network`, `app`. It becomes the
`index` argument in every tool call.

**Verify:**

```bash
./.venv/bin/prod-rag index list
./.venv/bin/prod-rag index status <short-name>
```

Expected: one row with a file/entity/chunk count, and `state fresh` with a
per-folder hash table. If it says `0 files`, the path is wrong or the subtree has
no C/C++ files — report that instead of guessing.

Useful flags when a subtree contains vendored code:

```bash
./.venv/bin/prod-rag index add core <path> --store neo4j \
    --exclude third_party --exclude build
```

---

## 4. Sanity-check retrieval before wiring anything up

Run these two and read the output. The first should return chunks and ontology
facts from the submodule you indexed; the second should print a full entity
record.

```bash
./.venv/bin/prod-rag search "<a question a new developer would ask>" --store neo4j
./.venv/bin/prod-rag entity <a symbol you know exists, e.g. Class::method> --store neo4j
```

**Verify:** the search output contains a `[FACT]` line and at least one code
chunk with `file:line`. The entity output contains `"found": true`.

If both are empty, the index is fine but the query does not match anything —
try a symbol name from the file list. If the entity is genuinely absent, the
submodule was indexed but that symbol lives elsewhere; report it.

---

## 5. Register the MCP server with your harness

The server command is:

```
<repo>/.venv/bin/prod-rag   mcp
```

It speaks MCP over **stdio** by default. Add it with whatever mechanism your
harness provides. Canonical JSON (`examples/mcp-config.json`):

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

Notes:

- Use the **absolute path** to `.venv/bin/prod-rag`; a bare `prod-rag` will not
  resolve in a harness that does not inherit your shell PATH.
- Do not add `--index` unless you deliberately want to restrict which indexes are
  visible. By default the server loads every built index.
- If your harness wants a `cwd`, leave it at the repo root.
- Optional env: `PROD_RAG_DATA` (index location, defaults to
  `~/.local/share/prod-rag`).

**Verify** by calling the tools from your harness (or with a script):

1. `list_indexes()` → returns the index you built in step 2.
2. `find_symbol("SomeKnownName")` → returns at least one match.
3. `get_entity("SomeKnownName")` → returns `"found": true` with `callers`,
   `throw_sites` and `fact_card`.

If the tools do not appear: run the command by hand in a terminal. It should
print nothing and wait for stdin. Any output before the handshake breaks the
protocol — capture the stderr and report it.

### Alternative: one shared HTTP instance

If your team already runs a service host:

```bash
<repo>/.venv/bin/prod-rag mcp --transport http --host 0.0.0.0 --port 8765
```

and register `{"type": "http", "url": "http://<host>:8765/mcp"}`. This has no
built-in authentication — do not expose it beyond a trusted network.

---

## 6. Optional: the web UI

```bash
<repo>/.venv/bin/prod-rag serve --port 8088
# with the Q&A box:
PROD_RAG_LLM_BASE_URL=http://<model-host>:<port> <repo>/.venv/bin/prod-rag serve --port 8088
```

Open `http://127.0.0.1:8088`. Ask/Browse/Indexes are the three views; the
Indexes view is where folder hashes and staleness appear.

**Verify:** the page loads, the header lists your index, and the Indexes view
shows a per-folder table with `fresh` rows.

---

## 7. How to use the tools yourself from now on

1. Call `list_indexes()` once to see what exists.
2. `search_code("<question>")` for anything about behaviour or location.
   It returns linked entities, exact `[FACT]` cards, and source chunks.
3. `get_entity("<qname>")` when you need one symbol's full picture: signature,
   callers, callees, fields initialised, inheritance, exception sites.
4. `get_neighbours("<qname>", hops=1)` when the question crosses symbols.
5. `read_chunk("<id>")` to read the exact source the index stored.

**Write the answer yourself from that evidence.** These tools deliberately do not
generate prose — they return facts. A tool-generated answer is available
(`answer_question`) only if a model endpoint was configured, and it is a weaker
model than you.

### After you edit code

The index is a snapshot. Before trusting an answer about files you changed:

```text
index_status("<name>")      -> state: stale, with the changed files
reindex("<name>")           -> rebuild (touches only the index, never the source)
```

`reindex` re-parses the whole indexed subtree and, with `PROD_RAG_STORE=neo4j`,
re-pushes it to the graph. Fractions of a second for a small submodule, ~31 s for
500 files / 185k lines. Not incremental.

---

## 8. Adding another submodule later

```bash
prod-rag index add <name> <path>
```

The running MCP server loaded its indexes at startup; restart it (or the harness
will reconnect on next start) to pick up the new one. The web UI has a **Rebuild**
button per index, but new indexes also need a restart.

---

## 9. What to report back

- The exact commands you ran, in order.
- `prod-rag doctor` output.
- `prod-rag neo4j status` output (uri + counts).
- The index name(s), root path(s), and the file/entity/chunk counts.
- Whether the three MCP verification calls in step 4 succeeded, quoting one
  result.
- Anything that failed, with the error text — do not paraphrase it.
- Anything you had to change (paths, ports, excludes) and why.

---

## Boundaries

- Never run `sudo pip install`, never modify the system Python, never disable
  the virtualenv.
- Never point `NEO4J_URI` at a shared or production database. `neo4j push`
  wipes `:Entity`/`:Chunk` unless `--keep` is given.
- This tool reads source code and writes indexes. If a step appears to require
  editing the codebase, stop — that is not part of this setup.
- Do not index a directory the human did not name.
- Do not commit anything; the repository is cloned, not authored, here.
