# -*- coding: utf-8 -*-
"""
index.py — build, store, inspect and reload codebase indexes.

An *index* is a named snapshot of one directory subtree: the parsed ontology
(entities + relations), the code chunks, and the content hashes of every folder
that contributed to it. Indexes are plain JSON on disk, so a colleague can copy
one, diff one, or inspect it with `jq`.

Why per-folder hashes
---------------------
Work codebases are huge. You index a submodule, not the world. But code moves,
so the index has to be able to answer three questions quickly:

  * which folders are in this index?
  * has anything in this folder changed since the index was built?
  * if yes, what exactly — added, removed, modified?

Hashes are content-based (sha256 over sorted `path:filehash` pairs), and each
folder carries a *subtree* hash, so a change three levels down marks every
ancestor stale. That is what the UI shows and what an agent can check before
trusting an answer.

Storage layout
--------------
    <data_dir>/indexes/<name>/index.json     the snapshot
    <data_dir>/indexes/<name>/meta.json      small record used by the registry

`data_dir` defaults to $XDG_DATA_HOME/prod-rag (~/.local/share/prod-rag).
Indexes are not in the repository: they describe *other* repositories.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from collections import Counter

from . import core as oc

CPP_EXTS = tuple(oc.CPP_EXTS) if hasattr(oc, "CPP_EXTS") else (
    ".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx", ".c", ".hpp")

# Directories that never contain hand-written project code worth indexing.
DEFAULT_EXCLUDE = (
    ".git", ".hg", ".svn", "build", "cmake-build-debug", "cmake-build-release",
    "node_modules", ".venv", "venv", "__pycache__", ".cache", "third_party",
    "3rdparty", "external", "dist", ".idea", ".vscode", "target", "out",
    # compiler and Qt output trees. Indexing them is worse than useless: the
    # generated moc_/ui_ classes shadow the real ones, and a class name then
    # resolves to a body-less stub (seen on a real 2050-file submodule).
    # Kept deliberately short: excluding a *source* directory by guesswork
    # would silently drop real code, which is worse than indexing a build tree.
    "obj", "CMakeFiles", "automoc",
)

# Generated translation units, matched on the file name.
GENERATED_PREFIXES = ("moc_", "ui_", "qrc_")
GENERATED_MARKERS = ("_generated.", ".pb.", ".pb-c.", "_automoc.cpp")


# ---------------------------------------------------------------------------
# data dir
# ---------------------------------------------------------------------------

def data_dir() -> str:
    base = os.environ.get("PROD_RAG_DATA") or os.environ.get("XDG_DATA_HOME") \
        or os.path.expanduser("~/.local/share")
    # PROD_RAG_DATA may point either at the base or at the prod-rag dir itself
    path = base if os.path.basename(base.rstrip("/")) == "prod-rag" \
        else os.path.join(base, "prod-rag")
    os.makedirs(os.path.join(path, "indexes"), exist_ok=True)
    return path


def index_dir(name: str) -> str:
    return os.path.join(data_dir(), "indexes", name)


def valid_name(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c in "-_." for c in name)


# ---------------------------------------------------------------------------
# scanning + hashing
# ---------------------------------------------------------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return _sha256_bytes(s.encode("utf-8", "surrogatepass"))


def scan(root: str, exts=CPP_EXTS, exclude=DEFAULT_EXCLUDE, follow_symlinks=False):
    """Walk `root` and return (files, dirs).

    files: {relpath: {"sha256", "size", "mtime"}}
    dirs : {reldir: subtree_hash}   ("" is the root; hashes include children)
    """
    root = os.path.abspath(root)
    exts = tuple(e.lower() for e in exts)
    exset = set(exclude)
    files: dict[str, dict] = {}

    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = sorted(d for d in dirnames if d not in exset and not d.startswith("."))
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        for fn in sorted(filenames):
            if not fn.lower().endswith(exts):
                continue
            low = fn.lower()
            if low.startswith(GENERATED_PREFIXES) or any(m in low for m in GENERATED_MARKERS):
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
                if st.st_size > 4 * 1024 * 1024:      # generated blobs are not code
                    continue
                with open(full, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            rel = os.path.join(rel_dir, fn) if rel_dir else fn
            files[rel.replace(os.sep, "/")] = {
                "sha256": _sha256_bytes(data),
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            }

    # subtree hashes, computed bottom-up over the directory tree
    tree: dict[str, dict] = {}

    def node_for(path: str) -> dict:
        if path not in tree:
            tree[path] = {"files": {}, "children": {}}
        return tree[path]

    for rel, meta in files.items():
        parts = rel.split("/")
        d = "/".join(parts[:-1])
        node_for(d)["files"][parts[-1]] = meta["sha256"]
        # make sure every ancestor exists
        while d:
            parent = d.rsplit("/", 1)[0] if "/" in d else ""
            node_for(parent)["children"][d] = True
            d = parent
    node_for("")

    dirs: dict[str, str] = {}

    def hash_node(path: str) -> str:
        node = tree.get(path, {"files": {}, "children": {}})
        lines = [f"f:{k}:{v}" for k, v in sorted(node["files"].items())]
        for child in sorted(node["children"]):
            lines.append(f"d:{os.path.basename(child)}:{hash_node(child)}")
        h = _sha256_text("\n".join(lines))
        dirs[path] = h
        return h

    hash_node("")
    return files, dirs


def _direct_files(files: dict, rel_dir: str) -> list[str]:
    if not rel_dir:
        return sorted(p for p in files if "/" not in p)
    pref = rel_dir + "/"
    return sorted(p for p in files if p.startswith(pref) and "/" not in p[len(pref):])


def _subtree_files(files: dict, rel_dir: str) -> list[str]:
    if not rel_dir:
        return sorted(files)
    pref = rel_dir + "/"
    return sorted(p for p in files if p.startswith(pref))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(name: str, root: str, exts=CPP_EXTS, exclude=DEFAULT_EXCLUDE,
          label: str | None = None, verbose: bool = True,
          with_retriever: bool = True) -> dict:
    """Parse `root` and write the index under `name`."""
    if not valid_name(name):
        raise ValueError(f"invalid index name {name!r} (letters, digits, - _ . only)")
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise ValueError(f"not a directory: {root}")

    t0 = time.time()
    files, dirs = scan(root, exts=exts, exclude=exclude)
    if not files:
        raise ValueError(
            f"no C/C++ files under {root} "
            f"(extensions {', '.join(exts)}; exclusions {', '.join(exclude)})")

    # One walk, one truth: the ontology is built from exactly the files that
    # `scan()` hashed, so freshness can never disagree with what was parsed.
    paths = sorted(files)
    if len(paths) > 1:
        onto = oc.build_from_files(paths, root=root, with_retriever=with_retriever)
    elif paths:
        onto = oc.build_from_file(os.path.join(root, paths[0]))
        if not with_retriever:
            onto._bm25 = None
    else:                                                    # pragma: no cover
        raise ValueError(f"no C/C++ files under {root}")
    if with_retriever:
        onto.build_retriever()      # BM25 over every chunk; skipped for Neo4j,
                                    # where Lucene does the lexical work
    build_ms = int((time.time() - t0) * 1000)

    payload = {
        "format": 1,
        "name": name,
        "label": label or os.path.basename(root) or root,
        "root": root,
        "exts": list(exts),
        "exclude": list(exclude),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_ms": build_ms,
        "files": files,
        "dirs": dirs,
        "stats": {
            "files": len(files),
            "dirs": len(dirs),
            "entities": len(onto.entities),
            "relations": len(onto.relations),
            "chunks": len(onto.chunks),
            "kinds": dict(Counter(e.kind for e in onto.entities.values())),
        },
        "entities": [{**dataclasses.asdict(e), "fact_card": onto._card(e)}
                     for e in onto.entities.values()],
        "relations": [dataclasses.asdict(r) for r in onto.relations],
        "chunks": onto.chunks,
    }

    out = index_dir(name)
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "index.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(out, "index.json"))

    meta = {k: payload[k] for k in
            ("format", "name", "label", "root", "exts", "exclude",
             "created_at", "updated_at", "build_ms", "stats")}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if verbose:
        s = payload["stats"]
        print(f"indexed {name!r}: {s['files']} files, {s['dirs']} dirs, "
              f"{s['entities']} entities, {s['relations']} relations, "
              f"{s['chunks']} chunks in {build_ms} ms")
    return meta


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def list_indexes() -> list[dict]:
    root = os.path.join(data_dir(), "indexes")
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        meta_path = os.path.join(root, name, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
    return out


def remove(name: str) -> bool:
    d = index_dir(name)
    idx = os.path.join(d, "index.json")
    if not os.path.exists(idx):
        return False
    os.replace(idx, idx + ".removed")
    for fn in ("meta.json",):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            os.remove(p)
    return True


def exists(name: str) -> bool:
    return os.path.exists(os.path.join(index_dir(name), "index.json"))


# ---------------------------------------------------------------------------
# loaded index: ontology + retrieval, ready to answer
# ---------------------------------------------------------------------------

class LoadedIndex:
    """One index, opened lazily.

    Only `meta.json` is read up front. The big `index.json` (tens of MB for a
    40k-entity submodule) is opened on first actual use, so an MCP server that
    exposes eight indexes starts in milliseconds instead of tens of seconds and
    pays the parse cost only for the index a question actually touches.
    """

    def __init__(self, name: str, store: "object | None" = None,
                 meta: dict | None = None, payload: dict | None = None):
        self._store = store          # GraphStore; defaults to MemoryStore
        self._meta = meta or {}
        self._payload = payload
        self.name = name
        self.label = self._meta.get("label") or name
        self.root = self._meta.get("root") or ""
        self.built_at = (self._meta.get("updated_at")
                         or self._meta.get("created_at"))
        self._onto: oc.CppOntology | None = None
        self._fresh: dict | None = None

    # ---- lazy payload -----------------------------------------------------
    @property
    def payload(self) -> dict:
        if self._payload is None:
            path = os.path.join(index_dir(self.name), "index.json")
            with open(path, encoding="utf-8") as f:
                self._payload = json.load(f)
            if not self._meta:
                self._meta = {k: self._payload.get(k) for k in
                              ("label", "root", "stats", "updated_at",
                               "created_at", "exts", "exclude")}
                self.label = self._meta.get("label") or self.name
                self.root = self._meta.get("root") or ""
                self.built_at = (self._meta.get("updated_at")
                                 or self._meta.get("created_at"))
        return self._payload

    @property
    def stats(self) -> dict:
        return self._meta.get("stats") or self.payload.get("stats", {})

    @property
    def files(self) -> dict:
        return self.payload.get("files", {})

    @property
    def dirs(self) -> dict:
        return self.payload.get("dirs", {})

    @property
    def exts(self) -> tuple:
        return tuple(self._meta.get("exts") or self.payload.get("exts") or CPP_EXTS)

    @property
    def exclude(self) -> tuple:
        return tuple(self._meta.get("exclude") or self.payload.get("exclude")
                     or DEFAULT_EXCLUDE)

    # ---- ontology ---------------------------------------------------------
    @property
    def onto(self) -> oc.CppOntology:
        if self._onto is None:
            onto = oc.CppOntology(self.label)
            fields = {f for f in oc.Entity.__dataclass_fields__}
            onto.entities = {
                e["id"]: oc.Entity(**{k: v for k, v in e.items() if k in fields})
                for e in self.payload["entities"]}
            onto.relations = [oc.Relation(**r) for r in self.payload["relations"]]
            onto.chunks = self.payload["chunks"]
            onto.files = sorted({e.file for e in onto.entities.values() if e.file})
            onto.index_names()       # the JSON payload has no name index
            if self._store is None or getattr(self._store, "name", "") != "neo4j":
                onto.build_retriever()
            self._onto = onto
        return self._onto

    # ---- freshness --------------------------------------------------------
    def freshness(self, rescan: bool = True) -> dict:
        """Compare the folders on disk with the folders recorded at build time."""
        if self._fresh is not None and not rescan:
            return self._fresh
        if not os.path.isdir(self.root):
            result = {
                "state": "missing",
                "detail": f"indexed root no longer exists: {self.root}",
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "folders": [],
                "changed": [], "added": [], "removed": [],
            }
            self._fresh = result
            return result

        cur_files, cur_dirs = scan(self.root, exts=self.exts, exclude=self.exclude)
        old_files, old_dirs = self.files, self.dirs

        changed, added, removed = [], [], []
        for rel, meta in cur_files.items():
            old = old_files.get(rel)
            if old is None:
                added.append(rel)
            elif old["sha256"] != meta["sha256"]:
                changed.append(rel)
        for rel in old_files:
            if rel not in cur_files:
                removed.append(rel)

        folders = []
        all_dirs = sorted(set(old_dirs) | set(cur_dirs))
        for d in all_dirs:
            old_hash = old_dirs.get(d)
            new_hash = cur_dirs.get(d)
            direct = _direct_files(old_files, d)
            if old_hash is None:
                status = "new"
            elif new_hash is None:
                status = "gone"
            elif old_hash != new_hash:
                status = "stale"
            else:
                status = "fresh"
            folders.append({
                "path": d or ".",
                "status": status,
                "files": len(direct),
                "subtree_files": len(_subtree_files(old_files, d)),
                "hash": (old_hash or "")[:16],
                "hash_now": (new_hash or "")[:16],
            })

        stale_any = bool(changed or added or removed)
        result = {
            "state": "stale" if stale_any else "fresh",
            "detail": (f"{len(changed)} modified, {len(added)} added, "
                       f"{len(removed)} removed") if stale_any else "index matches the working tree",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "built_at": self.built_at,
            "changed": changed[:500],
            "added": added[:500],
            "removed": removed[:500],
            "folders": folders,
        }
        self._fresh = result
        return result

    # ---- storage backend --------------------------------------------------
    @property
    def store(self):
        """The graph backend answering this index (memory by default)."""
        if self._store is None:
            from .store import MemoryStore
            self._store = MemoryStore(self.onto)
        return self._store

    # ---- retrieval --------------------------------------------------------
    def search(self, query: str, top_k: int = 8, hops: int = 1) -> dict:
        return self.store.search(query, top_k=top_k, hops=hops)

    def entity(self, name: str) -> dict | None:
        return self.store.entity(name)

    def _resolve(self, name: str) -> str | None:
        """Resolve a name to an entity id (used by read_chunk fallbacks)."""
        hit = self.store.entity(name)
        return hit["id"] if hit else None

    def neighbours(self, name: str, hops: int = 1) -> dict:
        return self.store.neighbours(name, hops=hops)

    def find(self, pattern: str, limit: int = 50) -> list[dict]:
        return self.store.find(pattern, limit=limit)

    def chunk(self, chunk_id: str) -> dict | None:
        return self.store.chunk(chunk_id)

    def answer(self, question: str, chat, top_k: int = 5, max_chars: int = 6000) -> dict:
        """Full path: retrieve -> generate -> verify -> (correct). Needs a model."""
        from .llm import answer_with_pipeline
        return answer_with_pipeline(self, question, chat, top_k=top_k,
                                    max_chars=max_chars)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(name: str, store=None) -> LoadedIndex:
    path = os.path.join(index_dir(name), "index.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no index named {name!r} (see `prod-rag index list`)")
    meta_path = os.path.join(index_dir(name), "meta.json")
    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = None
    return LoadedIndex(name, store=store, meta=meta)


def load_all(names: list[str] | None = None) -> dict[str, LoadedIndex]:
    metas = list_indexes()
    if names:
        wanted = [m["name"] for m in metas if m["name"] in set(names)]
        missing = set(names) - set(wanted)
        if missing:
            raise FileNotFoundError(f"unknown index(es): {', '.join(sorted(missing))}")
        metas = [m for m in metas if m["name"] in set(wanted)]
    return {m["name"]: load(m["name"]) for m in metas}
