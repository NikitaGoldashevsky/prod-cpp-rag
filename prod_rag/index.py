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
)


# ---------------------------------------------------------------------------
# data dir
# ---------------------------------------------------------------------------

def data_dir() -> str:
    base = os.environ.get("PROD_RAG_DATA") or os.environ.get("XDG_DATA_HOME") \
        or os.path.expanduser("~/.local/share")
    path = os.path.join(base, "prod-rag")
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
          label: str | None = None, verbose: bool = True) -> dict:
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

    onto = oc.build_from_dir(root) if len(files) > 1 else \
        oc.build_from_file(os.path.join(root, next(iter(files))))
    onto.build_retriever()
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
        "entities": [dataclasses.asdict(e) for e in onto.entities.values()],
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
    """One index, opened: ontology in memory, BM25 built, freshness known."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.name = payload["name"]
        self.label = payload.get("label") or self.name
        self.root = payload["root"]
        self.files: dict = payload.get("files", {})
        self.dirs: dict = payload.get("dirs", {})
        self.stats: dict = payload.get("stats", {})
        self.built_at = payload.get("updated_at") or payload.get("created_at")
        self._onto: oc.CppOntology | None = None
        self._fresh: dict | None = None

    # ---- ontology ---------------------------------------------------------
    @property
    def onto(self) -> oc.CppOntology:
        if self._onto is None:
            onto = oc.CppOntology(self.label)
            onto.entities = {e["id"]: oc.Entity(**e) for e in self.payload["entities"]}
            onto.relations = [oc.Relation(**r) for r in self.payload["relations"]]
            onto.chunks = self.payload["chunks"]
            onto.files = sorted({e.file for e in onto.entities.values() if e.file})
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

        cur_files, cur_dirs = scan(self.root,
                                   exts=tuple(self.payload.get("exts") or CPP_EXTS),
                                   exclude=tuple(self.payload.get("exclude") or DEFAULT_EXCLUDE))
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

    # ---- retrieval --------------------------------------------------------
    def search(self, query: str, top_k: int = 8, hops: int = 1) -> dict:
        onto = self.onto
        res = onto.retrieve(query, top_k=top_k, hops=hops)
        facts = res["facts"]
        if not facts:
            # A question that names no entity ("where are exceptions thrown
            # unguarded?") links nothing, but the chunks it still finds carry
            # fact cards. Return those rather than no ontology evidence at all.
            cards, seen = [], set()
            for ch, _ in res["chunks"][:3]:
                eid = ch["id"]
                if eid in onto.entities and eid not in seen:
                    seen.add(eid)
                    cards.append(onto._card(onto.entities[eid]))
            facts = "\n\n".join(cards)
        return {
            "index": self.name,
            "query": query,
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

    def entity(self, name: str) -> dict | None:
        onto = self.onto
        eid = self._resolve(name)
        if eid is None:
            return None
        return self._entity_payload(eid)

    def _resolve(self, name: str) -> str | None:
        onto = self.onto
        if name in onto.entities:
            return name
        low = name.lower()
        exact = [eid for eid, e in onto.entities.items()
                 if (e.qname or "").lower() == low or e.name.lower() == low]
        if exact:
            # prefer a definition over a declaration
            exact.sort(key=lambda eid: (not onto.entities[eid].is_definition,
                                        onto.entities[eid].kind in ("Parameter", "File")))
            return exact[0]
        sub = [eid for eid, e in onto.entities.items()
               if low in (e.qname or e.name).lower()]
        if sub:
            sub.sort(key=lambda eid: len(onto.entities[eid].qname or onto.entities[eid].name))
            return sub[0]
        return None

    def _entity_payload(self, eid: str) -> dict:
        onto = self.onto
        e = onto.entities[eid]
        calls = [onto.entities[r.o].qname if r.o in onto.entities else r.o
                 for r in onto.relations if r.s == eid and r.p == "calls"]
        refs = [onto.entities[r.o].qname if r.o in onto.entities else r.o
                for r in onto.relations if r.s == eid and r.p == "references"]
        uses = [r.o for r in onto.relations if r.s == eid and r.p == "callsExternal"]
        inits = [onto.entities[r.o].qname if r.o in onto.entities else r.o
                 for r in onto.relations if r.s == eid and r.p == "initializes"]
        inherited = [r.o for r in onto.relations if r.s == eid and r.p == "inheritsFrom"]
        inherited_by = [r.s for r in onto.relations
                        if r.o == eid and r.p == "inheritsFrom"]
        callers = [onto.entities[r.s].qname if r.s in onto.entities else r.s
                   for r in onto.relations if r.o == eid and r.p == "calls"]
        members = [onto.entities[r.o].qname if r.o in onto.entities else r.o
                   for r in onto.relations
                   if r.s == eid and r.p in ("hasMethod", "hasConstructor", "hasField")]
        chunk = next((c for c in onto.chunks if c["id"] == eid), None)
        return {
            "index": self.name,
            "id": eid,
            "kind": e.kind,
            "name": e.name,
            "qname": e.qname or e.name,
            "file": e.file,
            "line_start": e.line_start,
            "line_end": e.line_end,
            "is_definition": e.is_definition,
            "return_type": e.return_type,
            "type_ref": e.type_ref,
            "params": e.params,
            "modifiers": [k for k, v in
                          (("const", e.is_const), ("static", e.is_static),
                           ("virtual", e.is_virtual), ("noexcept", e.is_noexcept)) if v],
            "has_try_catch": e.has_try_catch,
            "throw_sites": e.throw_sites,
            "calls": sorted(set(calls)),
            "callers": sorted(set(callers)),
            "references": sorted(set(refs)),
            "uses": sorted(set(uses)),
            "initializes": inits,
            "inherits": inherited,
            "inherited_by": inherited_by,
            "members": sorted(set(members)),
            "fact_card": onto._card(e),
            "code": chunk["text"] if chunk else e.code,
        }

    def neighbours(self, name: str, hops: int = 1) -> dict:
        onto = self.onto
        eid = self._resolve(name)
        if eid is None:
            return {"index": self.name, "found": False, "name": name}
        ids = onto.neighbour_ids([eid], hops=max(1, min(hops, 3)))
        return {
            "index": self.name, "found": True, "root": eid,
            "entities": [{"id": i, "kind": onto.entities[i].kind,
                          "qname": onto.entities[i].qname or onto.entities[i].name,
                          "file": onto.entities[i].file}
                         for i in ids if i in onto.entities],
            "facts": onto.fact_cards(ids),
        }

    def find(self, pattern: str, limit: int = 50) -> list[dict]:
        """Glob-ish or substring search over entity names/qnames."""
        import fnmatch
        onto = self.onto
        pat = pattern.lower()
        globbed = any(ch in pattern for ch in "*?[]")
        hits = []
        for eid, e in onto.entities.items():
            target = (e.qname or e.name).lower()
            ok = fnmatch.fnmatch(target, pat) or fnmatch.fnmatch(e.name.lower(), pat) \
                if globbed else pat in target
            if ok and e.kind not in ("Parameter", "File", "GlobalBlock", "Header"):
                hits.append({"id": eid, "kind": e.kind,
                             "qname": e.qname or e.name,
                             "file": e.file, "line_start": e.line_start,
                             "is_definition": e.is_definition})
        hits.sort(key=lambda h: (len(h["qname"]), h["qname"]))
        return hits[:limit]

    def chunk(self, chunk_id: str) -> dict | None:
        onto = self.onto
        c = next((c for c in onto.chunks if c["id"] == chunk_id), None)
        if c is None:
            return None
        return {"index": self.name, **c}

    def answer(self, question: str, chat, top_k: int = 5, max_chars: int = 6000) -> dict:
        """Full path: retrieve -> generate -> verify -> (correct). Needs a model."""
        from .llm import answer_with_pipeline
        return answer_with_pipeline(self, question, chat, top_k=top_k,
                                    max_chars=max_chars)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(name: str) -> LoadedIndex:
    path = os.path.join(index_dir(name), "index.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no index named {name!r} (see `prod-rag index list`)")
    with open(path, encoding="utf-8") as f:
        return LoadedIndex(json.load(f))


def load_all(names: list[str] | None = None) -> dict[str, LoadedIndex]:
    metas = list_indexes()
    if names:
        wanted = [m["name"] for m in metas if m["name"] in set(names)]
        missing = set(names) - set(wanted)
        if missing:
            raise FileNotFoundError(f"unknown index(es): {', '.join(sorted(missing))}")
        metas = [m for m in metas if m["name"] in set(wanted)]
    return {m["name"]: load(m["name"]) for m in metas}
