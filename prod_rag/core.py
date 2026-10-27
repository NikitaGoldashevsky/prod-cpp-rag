# -*- coding: utf-8 -*-
"""
ontology_core.py — torch-free core of the ontology-augmented RAG pipeline for C++.

Responsibilities (all runnable on a CPU-only machine, no models needed):
  1. C++ parsing with tree-sitter (AST)
  2. Ontology extraction: entities (classes, methods, params, fields, types,
     headers, ...) + relations (hasMethod, hasParameter, calls, throws, includes,
     inheritsFrom, initializes, ...)
  3. Ontology-aware chunking (one chunk per AST entity, with metadata)
  4. A compact BM25 lexical retriever over chunks
  5. Entity linking + graph-neighbourhood expansion -> structured "fact cards"
     that are fed to the LLM together with the retrieved code chunks.

The generative/embedding half lives in the notebook; everything here is
deterministic and locally testable against cpp_test_code_from_practice.txt.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

# Vendored from the diploma prototype (`ontology_rag/ontology_core.py`) so this
# package is self-contained. Do not diverge silently: the retrieval behaviour is
# the thesis evidence. Changes belong in both repositories with a fresh parity run.

CPP_LANGUAGE = Language(tscpp.language())
_PARSER = Parser(CPP_LANGUAGE)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_QUAL  = re.compile(r"operator\s*[^\w\s]+")   # operator=, operator<< ...


def split_identifier(name: str) -> list[str]:
    """Split an identifier into meaningful sub-words (snake + camel aware).

    'hasActiveConnections_' -> ['has', 'active', 'connections']
    'distanceTo'            -> ['distance', 'to']
    """
    if not name:
        return []
    parts: list[str] = []
    for piece in re.split(r"[_]+", name):
        if not piece:
            continue
        parts.extend(_CAMEL.split(piece))
    return [p.lower() for p in parts if p and p != "_"]


def tokenize_code(text: str) -> list[str]:
    """Code-aware tokenization: identifiers split on punctuation AND on
    camelCase/snake_case; numeric literals and operators kept as symbols."""
    toks: list[str] = []
    # identifiers, numbers, operators/strings
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|\S", text):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
            toks.extend(split_identifier(raw))
            toks.append(raw.lower())
        else:
            toks.append(raw)
    return toks


def _text(src: bytes, node) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace").strip()


def _child_by_field(node, name):
    return node.child_by_field_name(name)


# ---------------------------------------------------------------------------
# BM25 (compact, dependency-free)
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.lengths = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.lengths) / self.N) if self.N else 0.0
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for d in docs_tokens:
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: __import__("math").log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        for t in query_tokens:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for i, counts in enumerate(self.tf):
                f = counts.get(t, 0)
                if not f:
                    continue
                dl = self.lengths[i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: str
    kind: str                     # Class, Struct, Method, Function, Constructor,
                                  # Destructor, Parameter, Field, Header, TypeAlias, File
    name: str
    qname: str = ""               # qualified name
    parent: str | None = None
    return_type: str | None = None
    type_ref: str | None = None   # for Parameter / Field
    params: list[dict] = field(default_factory=list)
    is_const: bool = False
    is_static: bool = False
    is_virtual: bool = False
    is_noexcept: bool = False
    access: str | None = None
    line_start: int = 0
    line_end: int = 0
    code: str = ""
    file: str = ""                 # source file this entity came from
    is_definition: bool = True      # False for prototypes / in-class declarations
    has_try_catch: bool = False
    throw_sites: list[dict] = field(default_factory=list)  # {type, guarded, line}


@dataclass
class Relation:
    s: str
    p: str
    o: str
    literal: bool = False


# ---------------------------------------------------------------------------
# C++ -> ontology
# ---------------------------------------------------------------------------

class CppOntology:
    def __init__(self, filename: str = "<source>"):
        self.filename = filename
        self.files: list[str] = []
        self._cur_file_id = f"file:{filename}"
        self._cur_stem = filename
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.chunks: list[dict] = []
        # index for name -> entity id (for call resolution / linking)
        self._by_name: dict[str, str] = {}
        self._by_qname: dict[str, list[str]] = {}
        self._cur_ns = ""      # id namespace of the file being processed
        self._cur_rel = ""     # relative path of the file being processed
        self._cur_dir = ""     # directory part of that path
        self._known_classes: set[str] = set()      # prescan: class/struct names
        self._known_namespaces: set[str] = set()   # prescan: namespace names
        self._out_index: dict | None = None
        self._adj: tuple | None = None
        # (caller_id, callee_text, n_args). The callee keeps its receiver
        # (`res.append`, not `append`) — dropping it is what let a QString
        # method call collide with an unrelated `TextForm::append`.
        self._raw_calls: list[tuple[str, str, int]] = []
        self._raw_refs: list[tuple[str, str]] = []
        # caller_id -> {local variable name: declared type}
        self._locals: dict[str, dict[str, str]] = {}

    # ---- public -----------------------------------------------------------
    def prescan(self, source_text: str):
        """Record class/struct and namespace names before extraction, so a
        qualified definition like `config::load` can be classified as a
        namespace function rather than a method of a class `config`."""
        try:
            src = source_text.encode("utf-8")
            root = _PARSER.parse(src).root_node
        except Exception:
            return
        for n in _walk(root):
            if n.type in ("class_specifier", "struct_specifier"):
                nm = _child_by_field(n, "name")
                if nm is not None:
                    self._known_classes.add(_text(src, nm))
            elif n.type == "namespace_definition":
                nm = _child_by_field(n, "name")
                if nm is not None:
                    self._known_namespaces.add(_text(src, nm))

    def build(self, source_text: str, filename: str | None = None):
        """Single-file entry point (back-compatible)."""
        self.prescan(source_text)
        self._build_one(source_text, filename or self.filename)
        self._resolve_calls_and_refs()
        self._finalise_chunks()
        return self

    def add_file(self, source_text: str, filename: str, namespace: str | None = None):
        """Append another source file to the same ontology (multi-file mode).

        `namespace` defaults to the directory of `filename` (pass relative
        paths, e.g. from `build_from_dir`) so that a class declared in a header
        and defined in the `.cpp` beside it share one entity, while same-named
        classes in different directories stay distinct.
        """
        rel = filename.replace("\\", "/")
        if namespace is None:
            namespace = os.path.dirname(rel)
        self._build_one(source_text, filename, namespace)
        return self

    def finish(self, with_retriever: bool = True):
        """Resolve cross-file calls and rebuild chunks/index after all files.

        `with_retriever=False` skips the BM25 index (the expensive part on large
        trees) when the caller hands the graph to a database that does its own
        lexical search.
        """
        self._resolve_calls_and_refs()
        self._finalise_chunks()
        if with_retriever:
            self.build_retriever()
        return self

    def _build_one(self, source_text: str, filename: str, namespace: str = ""):
        self.filename = filename
        rel = filename.replace("\\", "/")
        stem = rel.split("/")[-1]
        self._cur_rel = rel
        self._cur_dir = os.path.dirname(rel)
        self._cur_ns = namespace
        self._cur_stem = stem
        self._cur_file_id = self._unique_id(f"file:{rel}")
        if rel not in self.files:
            self.files.append(rel)
        self.src = source_text
        self.src_bytes = source_text.encode("utf-8")
        root = _PARSER.parse(self.src_bytes).root_node

        file_ent = Entity(id=self._cur_file_id, kind="File", name=stem, file=rel)
        self._add(file_ent)

        global_block_nodes: list = []
        for node in root.children:
            if node.type == "preproc_include":
                self._handle_include(node)
            elif node.type == "alias_declaration":
                self._handle_alias(node)
            elif node.type in ("class_specifier", "struct_specifier"):
                self._flush_global(global_block_nodes)
                self._handle_class(node)
            elif node.type == "function_definition":
                self._flush_global(global_block_nodes)
                self._handle_function(node)
            elif node.type == "declaration":
                self._flush_global(global_block_nodes)
                self._handle_top_declaration(node)
            elif node.type == "namespace_definition":
                # recurse one level
                body = node.child_by_field_name("body")
                if body:
                    for c in body.children:
                        if c.type == "function_definition":
                            self._flush_global(global_block_nodes)
                            self._handle_function(c)
                        else:
                            global_block_nodes.append(c)
            else:
                global_block_nodes.append(node)
        self._flush_global(global_block_nodes)
        return self

    # ---- node handlers ----------------------------------------------------
    def _qid(self, prefix: str, name: str) -> str:
        """Namespaced entity id. In multi-file mode the file's directory is a
        prefix, so same-named classes in different folders get distinct ids:
        `class:config/Config` vs `class:server_app/config/Config`."""
        if self._cur_ns:
            return f"{prefix}:{self._cur_ns}/{name}"
        return f"{prefix}:{name}"

    def _add(self, ent: Entity):
        # A name can legitimately occur more than once. Resolve it without
        # silently merging unrelated entities:
        #  * class stub + real declaration           -> keep the richer one
        #  * declaration + out-of-line definition    -> keep the definition
        #  * genuine overloads (two definitions)      -> keep both (new id)
        #  * anything else                            -> keep both (new id)
        if ent.id in self.entities and self.entities[ent.id] is not ent:
            existing = self.entities[ent.id]
            merged = self._merge_dup(existing, ent)
            if merged is not None:
                self.entities[existing.id] = merged
                return
            ent.id = self._unique_id(ent.id)
        self.entities[ent.id] = ent
        if ent.name:
            self._by_name.setdefault(ent.name, ent.id)
        if ent.qname:
            self._by_qname.setdefault(ent.qname, []).append(ent.id)

    def _merge_dup(self, existing: Entity, ent: Entity):
        """Return the entity to keep when `ent` collides with `existing` at the
        same id, or None if they are genuinely distinct (overloads)."""
        if existing.qname and ent.qname and existing.qname != ent.qname:
            return None
        if existing.kind in ("Class", "Struct") and ent.kind in ("Class", "Struct"):
            if existing.name != ent.name:
                return None
            return ent if (ent.code and not existing.code) else existing
        callables = ("Method", "Function", "Constructor", "Destructor")
        if existing.kind in callables and ent.kind in callables:
            if existing.is_definition and ent.is_definition:
                return None                      # two real definitions -> overloads
            if ent.is_definition and not existing.is_definition:
                return ent                       # upgrade declaration -> definition
            if existing.is_definition and not ent.is_definition:
                return existing                  # keep the definition
            return existing                      # two declarations -> keep first
        if existing.kind == ent.kind == "Field":
            return ent if (ent.code and not existing.code) else existing
        if existing.kind == ent.kind == "Parameter":
            return existing
        return None

    def _unique_id(self, base: str) -> str:
        """Deterministic unique id (handles overloaded callables)."""
        if base not in self.entities:
            return base
        i = 2
        while f"{base}#{i}" in self.entities:
            i += 1
        return f"{base}#{i}"

    def _rel(self, s: str, p: str, o: str, literal: bool = False):
        self.relations.append(Relation(s, p, o, literal))
        self._out_index = None          # invalidate the adjacency caches
        self._adj = None

    def _out(self, subject: str, predicate: str) -> list[str]:
        """Objects of `subject --predicate--> ?`, from a cached adjacency map.

        `_card()` used to scan the whole relation list four to six times per
        entity. On 500 files / 185k lines that is O(entities x relations) and
        took minutes; this makes it O(1) per lookup after one build.
        """
        idx = self._out_index
        if idx is None:
            idx = {}
            for r in self.relations:
                idx.setdefault(r.s, {}).setdefault(r.p, []).append(r.o)
            self._out_index = idx
        return idx.get(subject, {}).get(predicate, [])

    def _adjacency(self):
        """(outgoing, incoming) maps over non-literal edges, cached.

        Graph traversal used to rescan every relation for every frontier node,
        which is quadratic in both relations and hop count.
        """
        if self._adj is None:
            out: dict[str, list[str]] = {}
            inc: dict[str, list[str]] = {}
            for r in self.relations:
                if r.literal:
                    continue
                out.setdefault(r.s, []).append(r.o)
                inc.setdefault(r.o, []).append(r.s)
            self._adj = (out, inc)
        return self._adj

    def _handle_include(self, node):
        path = None
        for c in node.children:
            if c.type == "string_literal":
                path = _text(self.src_bytes, c).strip('"')
            elif c.type == "system_lib_string":
                path = _text(self.src_bytes, c).strip("<>")
        if path:
            hid = f"header:{path}"
            if hid not in self.entities:
                self._add(Entity(id=hid, kind="Header", name=path))
            self._rel(self._cur_file_id, "includes", hid)

    def _handle_alias(self, node):
        # using json = nlohmann::json;
        name = None
        for c in node.children:
            if c.type == "type_identifier":
                name = _text(self.src_bytes, c)
        if name:
            aid = self._qid("alias", name)
            self._add(Entity(id=aid, kind="TypeAlias", name=name,
                             file=self._cur_rel, code=_text(self.src_bytes, node)))
            self._rel(self._cur_file_id, "declares", aid)
            self._rel(aid, "aliasOf", _text(self.src_bytes, node).split("=", 1)[-1].strip(" ;"),
                      literal=True)

    def _flush_global(self, nodes):
        if not nodes:
            return
        text = "\n".join(_text(self.src_bytes, n) for n in nodes)
        if text.strip():
            base = "globals" if not self._cur_ns else f"globals:{self._cur_ns}"
            gid = self._unique_id(base)
            self._add(Entity(id=gid, kind="GlobalBlock", name="global declarations",
                             file=self._cur_rel, code=text))
        nodes.clear()

    def _class_header_relations(self, cls_node, cls_id):
        # inheritance
        base = None
        for c in cls_node.children:
            if c.type == "base_class_clause":
                base = _text(self.src_bytes, c)
        if base:
            for m in re.finditer(r"(?:public|private|protected)?\s*([A-Za-z_]\w*(?:::\w+)*)", base):
                bn = m.group(1)
                if bn in ("public", "private", "protected"):
                    continue
                self._rel(cls_id, "inheritsFrom", bn, literal=True)

    def _handle_class(self, node):
        name_node = _child_by_field(node, "name")
        name = _text(self.src_bytes, name_node) if name_node else "<anonymous>"
        cid = self._qid("class", name)
        kind = "Class" if node.type == "class_specifier" else "Struct"
        ent = Entity(id=cid, kind=kind, name=name, qname=name,
                     file=self._cur_rel,
                     line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                     code=_text(self.src_bytes, node))
        self._add(ent)
        self._rel(self._cur_file_id, "declares", cid)
        self._class_header_relations(node, cid)

        body = _child_by_field(node, "body")
        if not body:
            return
        current_access = "private" if kind == "Class" else "public"
        for child in body.children:
            if child.type == "access_specifier":
                m = re.match(r"(public|private|protected)", _text(self.src_bytes, child))
                if m:
                    current_access = m.group(1)
            elif child.type == "field_declaration":
                # an in-class member: a method declaration (has a function
                # declarator) or a data field
                if _has_descendant(child, "function_declarator"):
                    self._handle_method_declaration(child, cid, name, current_access)
                else:
                    self._handle_field(child, cid, name, current_access)
            elif child.type == "function_definition":
                self._handle_function(child, owner_class=name, owner_id=cid,
                                      access=current_access)
            elif child.type == "declaration":
                # inline member function declaration or member variable
                if any(c.type == "function_declarator" for c in child.children) or \
                   _has_descendant(child, "function_declarator"):
                    self._handle_method_declaration(child, cid, name, current_access)
                else:
                    self._handle_field(child, cid, name, current_access)

    def _handle_field(self, node, cls_id, cls_name, access):
        # could declare several identifiers: int a, b;
        type_node = _child_by_field(node, "type")
        type_txt = _text(self.src_bytes, type_node) if type_node else ""
        decls = []
        try:
            decl_nodes = node.children_by_field_name("declarator")
        except AttributeError:  # very old tree-sitter
            decl_nodes = []
        for d in decl_nodes:
            nm = _field_name(self.src_bytes, d)
            if nm:
                decls.append(nm)
        if not decls:
            type_txt, decls = _split_declaration(self.src_bytes, node)
        for dname in decls:
            fid = f"{cls_id}#{dname}"
            self._add(Entity(id=fid, kind="Field", name=dname, qname=f"{cls_name}::{dname}",
                             parent=cls_id, type_ref=type_txt, access=access,
                             file=self._cur_rel,
                             line_start=node.start_point[0] + 1,
                             line_end=node.end_point[0] + 1,
                             code=_text(self.src_bytes, node)))
            self._rel(cls_id, "hasField", fid)

    def _handle_method_declaration(self, node, cls_id, cls_name, access):
        fd = _find_function_declarator(node)
        if fd is None:
            return
        name = _declarator_name(self.src_bytes, fd)
        if not name:
            return
        ret = _declarator_return_type(self.src_bytes, node, fd)
        ent = Entity(id=f"{cls_id}::{name}", kind=_callable_kind(name, cls_name), name=name,
                     qname=f"{cls_name}::{name}", parent=cls_id, return_type=ret,
                     file=self._cur_rel, is_definition=False,
                     params=_parse_params(self.src_bytes, fd), access=access,
                     is_virtual=_has_descendant(node, "virtual") or "virtual" in _text(self.src_bytes, node),
                     is_static="static" in _text(self.src_bytes, node).split("(", 1)[0],
                     line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                     code=_text(self.src_bytes, node))
        self._add(ent)
        self._rel(cls_id, "hasMethod", ent.id)

    def _resolve_class(self, name: str):
        """Find the class/struct entity for `name`, preferring one declared in
        the same directory (so a `.cpp` attaches to its header's class when two
        same-named classes exist in different folders). None if ambiguous."""
        ids = [i for i, e in self.entities.items()
               if e.kind in ("Class", "Struct") and e.name == name]
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        same = [i for i in ids
                if os.path.dirname(self.entities[i].file or "") == self._cur_dir]
        return same[0] if len(same) == 1 else None

    def _handle_function(self, node, owner_class=None, owner_id=None, access=None):
        fd = _find_function_declarator(node)
        if fd is None:
            return
        qname = _declarator_qualified_name(self.src_bytes, fd)
        scope, name = _split_qualified(qname)
        if owner_class and not scope:
            scope = owner_class
            qname = f"{owner_class}::{name}"
        # resolve class scope to entity (create stub class if we never saw a decl)
        parent_id = owner_id
        in_namespace = False
        if scope:
            cid = owner_id or self._resolve_class(scope)
            if cid is not None:
                parent_id = cid
            elif scope in self._known_namespaces and scope not in self._known_classes:
                # namespace-qualified free function (e.g. config::loadFromJson)
                in_namespace = True
                parent_id = None
                qname = f"{scope}::{name}"
            else:
                cid = self._qid("class", scope)
                if cid not in self.entities:
                    self._add(Entity(id=cid, kind="Class", name=scope, qname=scope,
                                     file=self._cur_rel))
                    self._rel(self._cur_file_id, "declares", cid)
                parent_id = cid
        ret = _declarator_return_type(self.src_bytes, node, fd)
        params = _parse_params(self.src_bytes, fd)
        is_ctor = bool(scope) and name == scope and not in_namespace
        is_dtor = name.startswith("~")
        kind = "Constructor" if is_ctor else ("Destructor" if is_dtor else
                                              ("Method" if scope and not in_namespace else "Function"))
        base = f"{parent_id}::{name}" if parent_id else self._qid("func", qname)
        ent = Entity(id=base, kind=kind, name=name, qname=qname, parent=parent_id,
                     return_type=ret, params=params, access=access, file=self._cur_rel,
                     is_const=_has_descendant(fd, "type_qualifier") and
                              "const" in _text(self.src_bytes, fd).rsplit(")", 1)[-1],
                     is_noexcept=_has_descendant(fd, "noexcept"),
                     line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                     code=_text(self.src_bytes, node))
        self._add(ent)
        uid = ent.id
        if parent_id:
            self._rel(parent_id, "hasMethod" if not is_ctor else "hasConstructor", uid)
        else:
            self._rel(self._cur_file_id, "declares", uid)

        # parameters as entities
        for p in params:
            pid = f"{uid}#{p['name'] or 'arg'}"
            self._add(Entity(id=pid, kind="Parameter", name=p["name"] or "arg",
                             parent=uid, type_ref=p["type"], file=self._cur_rel,
                             code=p["code"]))
            self._rel(uid, "hasParameter", pid)

        # constructor initializer list -> initializes fields
        body = _child_by_field(node, "body")
        init_list = None
        for c in node.children:
            if c.type == "field_initializer_list":
                init_list = c
        if init_list:
            for fi in init_list.children:
                if fi.type == "field_initializer":
                    fname = None
                    for c in fi.children:
                        if c.type == "field_identifier":
                            fname = _text(self.src_bytes, c)
                    if fname and parent_id:
                        fid = f"{parent_id}#{fname}"
                        if fid not in self.entities:
                            self._add(Entity(id=fid, kind="Field", name=fname,
                                             qname=f"{scope}::{fname}", parent=parent_id,
                                             file=self._cur_rel))
                            self._rel(parent_id, "hasField", fid)
                        self._rel(uid, "initializes", fid)

        # body: calls / throws / symbol references / control flow
        if body:
            self._extract_calls(body, uid)
            self._extract_refs(body, uid)
            stored = self.entities.get(uid)
            if stored is not None:
                stored.has_try_catch = _has_descendant(body, "try_statement")
                stored.throw_sites = _collect_throws(self.src_bytes, body)
                for site in stored.throw_sites:
                    self._rel(uid, "throws", site["type"], literal=True)

    def _handle_top_declaration(self, node):
        text = _text(self.src_bytes, node)
        fd = _find_function_declarator(node)
        if fd is not None:
            name = _declarator_name(self.src_bytes, fd)
            ent = Entity(id=self._qid("func", name or "?"), kind="Function",
                         name=name or "?", qname=name or "?",
                         return_type=_declarator_return_type(self.src_bytes, node, fd),
                         params=_parse_params(self.src_bytes, fd), file=self._cur_rel,
                         is_definition=False,
                         line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                         code=text)
            self._add(ent)
            self._rel(self._cur_file_id, "declares", ent.id)
        else:
            # global variable declaration -> put in global block chunk only
            pass

    # ---- calls / throws ---------------------------------------------------
    def _extract_calls(self, node, caller_id):
        # Local declarations give the receiver of `res.append(...)` a type.
        # Without this the receiver is just a name and any same-named method
        # anywhere in the corpus is a candidate.
        locals_ = self._locals.setdefault(caller_id, {})
        for desc in _walk(node):
            if desc.type == "declaration":
                tnode = _child_by_field(desc, "type")
                tname = _normalize_type(_text(self.src_bytes, tnode)) if tnode else ""
                if tname:
                    try:
                        decls = desc.children_by_field_name("declarator")
                    except AttributeError:              # pragma: no cover
                        decls = []
                    for d in decls:
                        nm = _field_name(self.src_bytes, d)
                        if nm:
                            locals_.setdefault(nm, tname)
            elif desc.type == "call_expression":
                fn = _child_by_field(desc, "function")
                if fn is None:
                    continue
                callee = _callee_text(self.src_bytes, fn)
                if not callee:
                    continue
                args = _child_by_field(desc, "arguments")
                nargs = len(args.named_children) if args is not None else -1
                self._raw_calls.append((caller_id, callee, nargs))

    def _extract_refs(self, node, caller_id):
        """Capture `&Class::method` / qualified symbol references that are not
        plain calls (e.g. the thread target in run())."""
        for desc in _walk(node):
            if desc.type == "qualified_identifier":
                self._raw_refs.append((caller_id, _text(self.src_bytes, desc)))

    def _resolve_calls_and_refs(self):
        # Suffix lookup table: "::callee" -> [ids]. Built once instead of
        # scanning every qualified name per unresolved call, which is O(calls x
        # entities) and dominates build time on large trees (measured: minutes
        # for 500 files / 185k lines, seconds after this).
        by_suffix: dict[str, list[str]] = {}
        for qname, idlist in self._by_qname.items():
            idx = qname.rfind("::")
            if idx != -1:
                by_suffix.setdefault(qname[idx + 2:], []).extend(idlist)

        # class -> member name -> [ids]; class -> base names. Resolution walks
        # scope first (own class, then bases), which is what C++ does, instead
        # of guessing from every same-named symbol in the corpus.
        members: dict[str, dict[str, list[str]]] = {}
        bases: dict[str, list[str]] = {}
        for r in self.relations:
            if r.p in ("hasMethod", "hasConstructor", "hasField"):
                e = self.entities.get(r.o)
                if e is not None:
                    members.setdefault(r.s, {}).setdefault(e.name, []).append(r.o)
            elif r.p == "inheritsFrom":
                bases.setdefault(r.s, []).append(r.o)
        classes_by_name: dict[str, list[str]] = {}
        for eid, e in self.entities.items():
            if e.kind in ("Class", "Struct"):
                classes_by_name.setdefault(e.name, []).append(eid)
        ctx = {"by_suffix": by_suffix, "members": members, "bases": bases,
               "classes": classes_by_name, "fields": self._field_types()}

        seen_calls: set[tuple[str, str]] = set()
        for caller, callee, nargs in self._raw_calls:
            local = self._resolve_call(callee, self.entities.get(caller), ctx,
                                       nargs=nargs)
            if local:
                self._rel(caller, "calls", local)
                seen_calls.add((caller, local))
            else:
                self._rel(caller, "callsExternal", callee, literal=True)

        for caller, ref in self._raw_refs:
            local = self._resolve_call(ref, self.entities.get(caller), ctx)
            # a `references` edge is redundant when the same pair already has a
            # `calls` edge; a set makes that check O(1) instead of a scan.
            if local and (caller, local) not in seen_calls:
                self._rel(caller, "references", local)

    # ---- call resolution --------------------------------------------------
    def _resolve_call(self, callee: str, caller=None, ctx: dict | None = None,
                      nargs: int = -1):
        """Resolve a callee the way C++ scoping does, innermost scope first.

        Order: own class member -> base-class member -> same-file function ->
        same-directory function -> a name that is unique in the whole corpus.
        A receiver we cannot type (`res.append` where `res` is unknown) is left
        unresolved on purpose: guessing from the bare method name is what made
        `res.append(...)` look like a call to `TextForm::append`.
        """
        caller_file = caller.file if caller is not None else None
        if ctx is None:
            ctx = self._resolution_context()
        members, bases = ctx["members"], ctx["bases"]

        callee = (callee or "").strip()
        if not callee:
            return None
        # A leading `::` is an explicit request for global scope:
        # `::connect(sockfd, ...)` is the libc call, not `TcpClient::connect`
        # (a real false self-edge on cct-app).
        global_scope = callee.startswith("::")
        if global_scope:
            callee = callee[2:]
        callee = _strip_template_args(callee)
        if not callee:
            return None

        # `this->x()` / `this.x()` -> the caller's own class
        for sep in ("->", "."):
            if callee.startswith("this" + sep):
                return self._member_of(caller.parent if caller else None,
                                       callee[len("this" + sep):],
                                       members, bases, nargs, ctx)

        # `obj.method()` / `ptr->method()`: resolve only when obj's type is known
        if "->" in callee or "." in callee:
            sep = "->" if "->" in callee else "."
            recv, _, name = callee.rpartition(sep)
            recv, name = recv.strip(), name.strip()
            if recv and name:
                cls = self._type_class(recv, caller, ctx)
                if cls:
                    hit = self._member_of(cls, name, members, bases, nargs, ctx)
                    if hit:
                        return hit
                return None          # unknown receiver: do not guess
            return None

        # `Class::method` / `ns::fn` — qualified: exact match first
        if "::" in callee:
            hit = self._pick_entity(self._by_qname.get(callee, []), caller_file)
            if hit:
                return hit
            owner, _, name = callee.rpartition("::")
            cls = self._class_for(owner, caller_file, ctx)
            if cls:
                hit = self._member_of(cls, name, members, bases, nargs, ctx)
                if hit:
                    return hit
            # namespace-qualified free function defined without its namespace
            return self._pick_entity(self._by_qname.get(name, []), caller_file)

        # bare name: own class scope first, then its bases — unless the call
        # asked for global scope explicitly
        if not global_scope:
            hit = self._member_of(caller.parent if caller else None, callee,
                                  members, bases, nargs, ctx)
            if hit:
                return hit

        # `Type(args)` is a constructor call. Without this, `return DeviceState(...)`
        # loses its edge entirely — the class name is neither a member nor a
        # free function.
        if callee in ctx["classes"]:
            cls = self._class_for(callee, caller_file, ctx)
            if cls:
                ctor = self._member_of(cls, callee.split("::")[-1], members,
                                       bases, nargs, ctx)
                return ctor or cls

        # free functions: same file, then same directory
        cands = [i for i in self._by_qname.get(callee, [])
                 if i in self.entities and self.entities[i].kind == "Function"]
        hit = self._pick_entity(cands, caller_file)
        if hit:
            return hit
        # Otherwise only a *free function* that is unambiguous corpus-wide. A
        # method of an unrelated class is never a valid target for a bare call:
        # `render()` inside SingleTimer is not `Widget::render`, it is either a
        # member (handled above) or a free function. Guessing here produced
        # exactly that false edge on a real 667-file submodule.
        uniq = {i for i in ctx["by_suffix"].get(callee, [])
                if i in self.entities and self.entities[i].kind == "Function"}
        if len(uniq) == 1:
            return next(iter(uniq))
        return None

    # ---- resolution helpers ----------------------------------------------
    def _resolution_context(self) -> dict:
        """Indexes `_resolve_call` needs: members, bases, classes, suffixes."""
        by_suffix: dict[str, list[str]] = {}
        for qname, idlist in self._by_qname.items():
            i = qname.rfind("::")
            if i != -1:
                by_suffix.setdefault(qname[i + 2:], []).extend(idlist)
        members: dict[str, dict[str, list[str]]] = {}
        bases: dict[str, list[str]] = {}
        for r in self.relations:
            if r.p in ("hasMethod", "hasConstructor", "hasField"):
                e = self.entities.get(r.o)
                if e is not None:
                    members.setdefault(r.s, {}).setdefault(e.name, []).append(r.o)
            elif r.p == "inheritsFrom":
                bases.setdefault(r.s, []).append(r.o)
        classes: dict[str, list[str]] = {}
        for eid, e in self.entities.items():
            if e.kind in ("Class", "Struct"):
                classes.setdefault(e.name, []).append(eid)
        return {"by_suffix": by_suffix, "members": members, "bases": bases,
                "classes": classes, "fields": self._field_types()}

    def _class_for(self, name: str, near_file: str | None, ctx: dict):
        """The class entity for `name`, preferring one from the same directory."""
        ids = [i for i in ctx["classes"].get(name, []) if i in self.entities]
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        if near_file:
            same_file = [i for i in ids if self.entities[i].file == near_file]
            if len(same_file) == 1:
                return same_file[0]
            cdir = os.path.dirname(near_file)
            same_dir = [i for i in ids
                        if os.path.dirname(self.entities[i].file or "") == cdir]
            if len(same_dir) == 1:
                return same_dir[0]
        return None

    def _member_of(self, class_id, name: str, members: dict, bases: dict,
                   nargs: int = -1, ctx: dict | None = None, depth: int = 0):
        """Look `name` up in a class and its bases (depth-limited)."""
        if not class_id or depth > 3 or class_id not in self.entities:
            return None
        ids = members.get(class_id, {}).get(name)
        if ids:
            return self._pick_overload(ids, nargs)
        near = self.entities[class_id].file
        for base_name in bases.get(class_id, []):
            base_id = self._class_for(base_name, near, ctx) if ctx else None
            if base_id and base_id != class_id:
                hit = self._member_of(base_id, name, members, bases, nargs,
                                      ctx, depth + 1)
                if hit:
                    return hit
        return None

    def _pick_overload(self, ids: list[str], nargs: int):
        """Choose between overloads by argument count, when that is decisive."""
        ids = [i for i in dict.fromkeys(ids) if i in self.entities]
        if not ids:
            return None
        if len(ids) == 1 or nargs < 0:
            return ids[0]
        exact = [i for i in ids if len(self.entities[i].params) == nargs]
        if exact:
            return exact[0]
        fits = [i for i in ids if len(self.entities[i].params) >= nargs]
        return fits[0] if fits else ids[0]

    def _type_class(self, expr: str, caller, ctx: dict):
        """The class a receiver expression has, if we can know it honestly."""
        expr = expr.strip()
        if expr in ("this", ""):
            return caller.parent if caller else None
        if expr[0] in "*&":
            expr = expr[1:].strip()
        cls_names = ctx["classes"]
        near = caller.file if caller else None
        if expr in cls_names:                # `Foo.bar()` / `Foo->bar()`
            return self._class_for(expr, near, ctx)
        tname = self._declared_type(expr, caller, ctx)
        if tname and tname in cls_names:
            return self._class_for(tname, near, ctx)
        return None

    def _declared_type(self, var: str, caller, ctx: dict | None = None):
        """Declared type of a parameter, a local, or a field of the caller."""
        if caller is None:
            return None
        for p in caller.params or []:
            if p.get("name") == var:
                return _normalize_type(p.get("type") or "")
        local = self._locals.get(caller.id, {}).get(var)
        if local:
            return local
        ctx = ctx or self._resolution_context()
        fields = ctx["fields"]
        cls = caller.parent
        for _ in range(4):                    # walk the base chain
            if not cls or cls not in self.entities:
                return None
            t = fields.get(cls, {}).get(var)
            if t:
                return t
            nb = ctx["bases"].get(cls, [])
            if not nb:
                return None
            cls = self._class_for(nb[0], caller.file, ctx)
        return None

    def _field_types(self) -> dict[str, dict[str, str]]:
        """class id -> {field name: normalized type}. Built once per build."""
        out: dict[str, dict[str, str]] = {}
        for r in self.relations:
            if r.p != "hasField":
                continue
            f = self.entities.get(r.o)
            if f is None:
                continue
            out.setdefault(r.s, {})[f.name] = _normalize_type(f.type_ref or "")
        return out

    def _caller_file(self, entity_id: str):
        e = self.entities.get(entity_id)
        return e.file if e else None

    def _pick_entity(self, ids, caller_file):
        """Disambiguate a set of same-named entities. Prefers the one defined in
        the caller's own file, then directory; returns None if still ambiguous."""
        ids = [i for i in dict.fromkeys(ids) if i in self.entities]
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        if caller_file:
            same_file = [i for i in ids if self.entities[i].file == caller_file]
            if len(same_file) == 1:
                return same_file[0]
            cdir = os.path.dirname(caller_file)
            same_dir = [i for i in ids
                        if os.path.dirname(self.entities[i].file or "") == cdir]
            if len(same_dir) == 1:
                return same_dir[0]
        return None

    def _extract_throws(self, node, caller_id):
        for site in _collect_throws(self.src_bytes, node):
            self._rel(caller_id, "throws", site["type"], literal=True)

    # ---- chunks -----------------------------------------------------------
    def _finalise_chunks(self):
        self.chunks = []
        # one chunk per function/method/ctor/dtor, in source order
        callables = [e for e in self.entities.values()
                     if e.kind in ("Method", "Function", "Constructor", "Destructor")]
        callables.sort(key=lambda e: (e.file or "", e.line_start))
        for e in callables:
            if e.code:
                self.chunks.append({
                    "id": e.id, "kind": e.kind, "qname": e.qname,
                    "file": e.file,
                    "line_start": e.line_start, "line_end": e.line_end,
                    "text": _chunk_with_context(self, e),
                })
        # field chunks
        for e in self.entities.values():
            if e.kind == "Field" and e.code:
                self.chunks.append({
                    "id": e.id, "kind": "Field", "qname": e.qname,
                    "file": e.file,
                    "line_start": e.line_start, "line_end": e.line_end,
                    "text": _chunk_with_context(self, e),
                })
        # class chunks (full text)
        for e in self.entities.values():
            if e.kind in ("Class", "Struct") and e.code:
                self.chunks.append({
                    "id": e.id, "kind": e.kind, "qname": e.qname,
                    "file": e.file,
                    "line_start": e.line_start, "line_end": e.line_end, "text": e.code,
                })
        # globals (one per file)
        for e in self.entities.values():
            if e.kind == "GlobalBlock" and e.code:
                self.chunks.append({"id": e.id, "kind": "GlobalBlock",
                                    "qname": e.name, "file": e.file,
                                    "line_start": e.line_start, "line_end": e.line_end,
                                    "text": e.code})

    # ---- fact cards -------------------------------------------------------
    def fact_cards(self, entity_ids: list[str]) -> str:
        out = []
        seen = set()
        for eid in entity_ids:
            e = self.entities.get(eid)
            if not e or eid in seen or e.kind in ("File", "GlobalBlock"):
                continue
            seen.add(eid)
            out.append(self._card(e))
        return "\n\n".join(out)

    def _card(self, e: Entity) -> str:
        if e.kind in ("Method", "Function", "Constructor", "Destructor"):
            sig = e.qname or e.name
            params = ", ".join(f"{p['type']} {p['name']}".strip() for p in e.params) or "void"
            bits = [f"{e.kind} {sig}({params})"]
            if e.return_type:
                bits.append(f"returns {e.return_type}")
            flags = [f for f, v in (("const", e.is_const), ("noexcept", e.is_noexcept),
                                    ("static", e.is_static), ("virtual", e.is_virtual)) if v]
            if flags:
                bits.append("modifiers: " + ", ".join(flags))
            if e.parent:
                bits.append(f"member of {self.entities[e.parent].name if e.parent in self.entities else e.parent}")
            calls = [self._label(o) for o in self._out(e.id, "calls")]
            refs = [self._label(o) for o in self._out(e.id, "references")]
            ext = self._out(e.id, "callsExternal")
            inits = [self._label(o) for o in self._out(e.id, "initializes")]
            if calls:
                bits.append("calls: " + ", ".join(sorted(set(calls))))
            if refs:
                bits.append("references: " + ", ".join(sorted(set(refs))))
            if ext:
                bits.append("uses: " + ", ".join(sorted(set(ext))))
            if e.throw_sites:
                parts = sorted({f"{s['type']}{'' if s['guarded'] else ' (unguarded, propagates)'}"
                                for s in e.throw_sites})
                bits.append("throws: " + ", ".join(parts))
            if e.has_try_catch:
                bits.append("has try/catch")
            elif calls or ext or e.throw_sites:
                bits.append("NO try/catch: exceptions from calls are NOT handled here")
            if inits:
                bits.append("initializes: " + ", ".join(inits))
            bits.append(f"lines {e.line_start}-{e.line_end}"
                        + (f" in {e.file}" if e.file else ""))
            return "[FACT] " + "; ".join(bits) + "."
        if e.kind == "Field":
            owner = self.entities[e.parent].name if e.parent in self.entities else e.parent
            t = f" of type {e.type_ref}" if e.type_ref else ""
            tail = f"; lines {e.line_start}-{e.line_end}." if e.line_start else "."
            return f"[FACT] Field {e.name}{t} belongs to {owner}{tail}"
        if e.kind == "Class" or e.kind == "Struct":
            methods = [self.entities[o].name
                       for o in self._out(e.id, "hasMethod") + self._out(e.id, "hasConstructor")
                       if o in self.entities]
            fields = [self.entities[o].name for o in self._out(e.id, "hasField")
                      if o in self.entities]
            bases = self._out(e.id, "inheritsFrom")
            bits = [f"{e.kind} {e.name}"]
            if bases:
                bits.append("inherits " + ", ".join(bases))
            if methods:
                bits.append("methods: " + ", ".join(sorted(set(methods))))
            if fields:
                bits.append("fields: " + ", ".join(sorted(set(fields))))
            return "[FACT] " + "; ".join(bits) + "."
        return f"[FACT] {e.kind} {e.qname or e.name}."

    def _label(self, eid: str) -> str:
        e = self.entities.get(eid)
        return e.qname if e else eid

    # ---- entity linking ---------------------------------------------------
    def link(self, query: str, threshold: float = 0.9, max_hits: int = 12) -> list[str]:
        """Find ontology entities mentioned in a query (word-bounded exact + fuzzy)."""
        q_tokens: set[str] = set()
        for w in re.findall(r"[A-Za-z_]\w*", query):
            q_tokens.update(split_identifier(w))
        q_words = {w.lower() for w in re.findall(r"[A-Za-z_]\w*", query)}
        q_low = query.lower()
        hits: dict[str, float] = {}
        for eid, e in self.entities.items():
            if e.kind in ("File", "GlobalBlock", "Parameter"):
                continue
            name = e.name or ""
            if not name:
                continue
            ln = len(name)
            qn = (e.qname or "").lower()
            score = 0.0
            # 1) exact token (covers short names like run, stop)
            if ln >= 2 and name.lower() in q_words:
                score = 1.0 + 0.01 * ln
            # 2) word-bounded substring of the entity name / qname
            if ln >= 3:
                pat = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name.lower())
                if re.search(pat, q_low):
                    score = max(score, 1.0 + 0.01 * ln)
            # 2b) all sub-tokens of the entity name appear in the query
            #     (links e.g. 'distanceTo' from "calculate the distance to ...")
            if score == 0.0 and ln >= 4:
                etoks = set(split_identifier(name))
                if etoks and etoks <= q_tokens:
                    score = 0.85
            if score == 0.0 and qn and len(qn) >= 3 and qn in q_low:
                score = 1.0 + 0.01 * len(qn)
            # 3) fuzzy only for reasonably long identifiers
            if score == 0.0 and ln >= 6:
                best = 0.0
                for w in q_words:
                    if abs(len(w) - ln) > max(2, ln // 3):
                        continue
                    best = max(best, SequenceMatcher(None, name.lower(), w).ratio())
                if best >= threshold:
                    score = best * 0.8
            if score > 0:
                hits[eid] = score
        ranked = sorted(hits.items(), key=lambda kv: -kv[1])
        # keep the best-scoring entity per id; distinct entities that merely
        # share a simple name (different namespaces) are kept separately
        out, seen = [], set()
        for eid, _ in ranked:
            if eid in seen:
                continue
            seen.add(eid)
            out.append(eid)
            if len(out) >= max_hits:
                break
        return out

    def neighbour_ids(self, entity_ids: list[str], hops: int = 1) -> list[str]:
        result = list(entity_ids)
        seen = set(result)
        frontier = list(entity_ids)
        for _ in range(hops):
            nxt = []
            out_map, in_map = self._adjacency()
            for nid in frontier:
                # outgoing (except callsExternal literals)
                for o in out_map.get(nid, ()):
                    if o in self.entities and o not in seen:
                        nxt.append(o); seen.add(o)
                for s_ in in_map.get(nid, ()):
                    if s_ in self.entities and s_ not in seen:
                        nxt.append(s_); seen.add(s_)
            frontier = nxt
            result.extend(nxt)
        return result

    # ---- retrieval --------------------------------------------------------
    def build_retriever(self):
        # Index each chunk together with its structured fact card, so that
        # question words like 'parameters' / 'returns' / 'throws' / 'calls'
        # match the ontology layer, not just the raw code text.
        self._chunk_tokens = []
        for c in self.chunks:
            card = self._card(self.entities[c["id"]]) if c["id"] in self.entities else ""
            self._chunk_tokens.append(tokenize_code(c["text"] + "\n" + card))
        self._bm25 = BM25(self._chunk_tokens)
        return self

    def lexical_search(self, query: str, top_k: int = 5):
        scores = self._bm25.get_scores(tokenize_code(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        return [(self.chunks[i], scores[i]) for i in order[:top_k] if scores[i] > 0]

    def retrieve(self, query: str, top_k: int = 5, hops: int = 1):
        """Ontology-primary retrieval: link entities, expand neighbourhood,
        then fill remaining budget with lexical code chunks."""
        linked = self.link(query)
        expanded = self.neighbour_ids(linked, hops=hops)
        facts = self.fact_cards(expanded)

        lex = self.lexical_search(query, top_k=top_k)
        # dedupe chunks: prefer chunks whose entity is in the ontology hit set
        chunks = []
        used = set()
        for ch, sc in lex:
            if ch["id"] in used:
                continue
            used.add(ch["id"])
            chunks.append((ch, sc))
        # also ensure a chunk exists for top linked callables
        for eid in linked:
            if eid in self.entities and self.entities[eid].kind in (
                    "Method", "Function", "Constructor", "Destructor", "Class", "Struct"):
                for ch in self.chunks:
                    if ch["id"] == eid and ch["id"] not in used:
                        used.add(ch["id"])
                        chunks.append((ch, 0.0))
        return {"linked": linked, "facts": facts, "chunks": chunks}

    def format_context(self, res, max_chars: int | None = None) -> str:
        """Render a retrieval result as the context block handed to the LLM."""
        lines = ["### ONTOLOGY FACTS (structured, exact):", "", res["facts"], "",
                 "### RELEVANT CODE CHUNKS:", ""]
        for ch, _ in res["chunks"]:
            fnote = f" [{ch['file']}]" if ch.get("file") else ""
            lines.append(f"// {ch['kind']} {ch['qname']}{fnote} "
                         f"(lines {ch['line_start']}-{ch['line_end']})")
            lines.append(ch["text"])
            lines.append("")
        text = "\n".join(lines)
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n... [context truncated]"
        return text

    # ---- export -----------------------------------------------------------
    def to_triples(self) -> list[tuple]:
        t = []
        for r in self.relations:
            if r.literal:
                obj = f'"{r.o}"'
            else:
                obj = f"<{r.o}>"
            t.append((f"<{r.s}>", f"cpp:{r.p}", obj))
        return t

    def to_rdf(self):
        """Optional: build a real RDF graph (rdflib)."""
        from rdflib import Graph, Namespace, URIRef, Literal
        from rdflib.namespace import RDF, RDFS
        g = Graph()
        CPP = Namespace("http://example.org/cpp#")
        g.bind("cpp", CPP)
        kinds = {"Class": CPP.Class, "Struct": CPP.Struct, "Method": CPP.Method,
                 "Function": CPP.Function, "Constructor": CPP.Constructor,
                 "Destructor": CPP.Destructor, "Field": CPP.Field,
                 "Parameter": CPP.Parameter, "Header": CPP.Header,
                 "TypeAlias": CPP.TypeAlias, "File": CPP.SourceFile,
                 "GlobalBlock": CPP.GlobalBlock}
        for e in self.entities.values():
            subj = URIRef(f"http://example.org/cpp#{e.id}")
            g.add((subj, RDF.type, kinds.get(e.kind, CPP.Entity)))
            g.add((subj, CPP.name, Literal(e.name)))
            if e.qname:
                g.add((subj, CPP.qualifiedName, Literal(e.qname)))
            if e.return_type:
                g.add((subj, CPP.returnType, Literal(e.return_type)))
            if e.type_ref:
                g.add((subj, CPP.hasType, Literal(e.type_ref)))
        for r in self.relations:
            s = URIRef(f"http://example.org/cpp#{r.s}")
            p = CPP[r.p]
            o = Literal(r.o) if r.literal else URIRef(f"http://example.org/cpp#{r.o}")
            g.add((s, p, o))
        return g


# ---------------------------------------------------------------------------
# Rank fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Combine several ranked id-lists into one order (RRF).

    Used to fuse the ontology-lexical ranking with the embedding ranking.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k + rank + 1)
    return [i for i, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------------------
# Grounding: verification prompts + deterministic guards
# ---------------------------------------------------------------------------

_VERIFY_INSTRUCTIONS = (
    "You are a careful fact-checker for answers about code.\n"
    "You are given a QUESTION, the CONTEXT that was available, and an ANSWER.\n"
    "Decide whether the CONTEXT supports the ANSWER's factual claims.\n"
    "A claim is supported if the context (ontology facts or code) states it, or if\n"
    "the answer only restates/quotes/summarises the context. Paraphrasing, quoting\n"
    "identifiers verbatim, and reasonable restatement of what the context says are\n"
    "all SUPPORTED - do not flag them.\n"
    "Only flag a claim as unsupported when it ADDS information that is NOT in the\n"
    "context, CONTRADICTS the context, or invents a value/number/flag/default.\n\n"
    "IMPORTANT: if the ANSWER says something is 'not specified', 'not found',\n"
    "'not defined' or otherwise absent, and the CONTEXT indeed does not provide it,\n"
    "that is SUPPORTED (an honest 'not specified' is never an unsupported claim).\n"
    "When in doubt and the answer is consistent with the context, reply SUPPORTED.\n\n"
    "Reply with EXACTLY one word on its own line: SUPPORTED or UNSUPPORTED.\n"
    "If UNSUPPORTED, list only the unsupported claims, one per line, prefixed '- '.\n"
    "Output nothing else.\n")


def build_verify_prompt(question: str, context: str, answer: str) -> str:
    return (f"{_VERIFY_INSTRUCTIONS}\nQUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context}\n\nANSWER TO CHECK:\n{answer}\n")


def build_correct_prompt(question: str, context: str, answer: str,
                         critique: list[str]) -> str:
    bullets = "\n".join("- " + c for c in critique) or "- (unspecified)"
    return (f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
            f"DRAFT ANSWER:\n{answer}\n\n"
            "A verifier found that these claims are NOT supported by the context:\n"
            f"{bullets}\n\n"
            "Rewrite the answer so that EVERY claim is directly supported by the context. "
            "Remove or correct the unsupported claims. If something is not stated in the "
            "context, say it is not specified. Output ONLY the corrected answer.\n")


_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Remove reasoning-model think blocks (including a dangling open/close tag)
    so raw `...`/`</think>` never leaks into an answer."""
    if not text:
        return ""
    out = _THINK_RE.sub("", text)
    # a lone stray tag (e.g. a truncated think block leaving '</think>')
    out = re.sub(r"</?think(?:ing)?>", "", out, flags=re.I)
    return out.strip()


def parse_verdict(text: str) -> tuple[bool, list[str]]:
    """Parse a (possibly chatty / markdown) verifier reply into
    (supported?, unsupported_claims).

    Robust to markdown (`**UNSUPPORTED**`), a 'Verdict:' prefix, leading
    reasoning, and the word appearing anywhere rather than on line 1.
    An empty/unparseable reply is treated as UNSUPPORTED (safe default).
    """
    if not text or not text.strip():
        return False, []
    norm = re.sub(r"[*_`#>]", " ", text)
    low = norm.lower()
    uns = re.search(r"\bunsupported\b|\bnot\s+supported\b", low)
    sup = re.search(r"\bsupported\b", low)
    if uns and (sup is None or uns.start() < sup.start()):
        ok, pos = False, uns.end()
    elif sup:
        ok, pos = True, sup.end()
    else:
        return False, []
    claims = []
    for line in norm[pos:].splitlines():
        s = line.strip().lstrip("-*\u2022 \t")
        if not s or s.startswith(":"):
            continue
        if s.lower() in ("verdict", "reason", "reasons"):
            continue
        claims.append(s)
    return ok, claims


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def suspicious_numbers(answer: str, context: str) -> list[str]:
    """Deterministic guard: numbers in the answer that never appear in the
    context (a cheap catch for invented values)."""
    return sorted({n for n in _NUM.findall(answer) if n not in context})


def verified_answer(query, context, draft_fn, verify_fn, correct_fn,
                    numbers_fn=suspicious_numbers):
    """Run the generate -> verify -> (correct -> re-verify) loop.

    The four I/O steps are injected, so the control logic is model-free and
    unit-testable:
      draft_fn(query, context)               -> str
      verify_fn(query, context, answer)      -> (ok: bool, claims: list[str])
      correct_fn(query, context, ans, crit)  -> str

    Guarantee: a correction is accepted only if it PASSES verification;
    otherwise the original draft is kept (the loop can never make it worse).
    """
    draft = draft_fn(query, context)
    ok, claims = verify_fn(query, context, draft)
    bad = numbers_fn(draft, context)
    if ok and not bad:
        return draft, {"supported": True, "claims": [], "bad_numbers": [], "corrected": False}

    critique = list(claims) + [f"the number {n} never appears in the context" for n in bad]
    fixed = correct_fn(query, context, draft, critique)
    ok2, _ = verify_fn(query, context, fixed)
    if ok2:
        return fixed, {"supported": True, "claims": claims, "bad_numbers": bad,
                       "corrected": True, "draft": draft}
    return draft, {"supported": bool(ok), "claims": claims, "bad_numbers": bad,
                   "corrected": True, "correction_rejected": True, "draft": draft}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)


def _has_descendant(node, type_name: str) -> bool:
    return any(d.type == type_name for d in _walk(node) if d is not node)


def _find_descendant(node, type_name: str):
    for d in _walk(node):
        if d.type == type_name:
            return d
    return None


def _collect_throws(src: bytes, node) -> list[dict]:
    """Throw sites inside a function body, tagged with whether they are
    guarded by a try/catch ('guarded' = an ancestor up to the body is a
    try_statement)."""
    sites = []
    for desc in _walk(node):
        if desc.type != "throw_statement":
            continue
        txt = _text(src, desc)
        m = re.search(r"throw\s+([A-Za-z_][\w:]*)\s*\(", txt) or \
            re.search(r"throw\s+([A-Za-z_][\w:]*)", txt)
        etype = m.group(1) if m else "<expression>"
        guarded = _has_ancestor(node, desc, "try_statement")
        sites.append({"type": etype, "guarded": guarded, "line": desc.start_point[0] + 1})
    return sites


def _has_ancestor(root, node, type_name: str) -> bool:
    """Walk parent links from node up to (excluding) root, checking node types."""
    cur = node.parent
    while cur is not None and cur is not root:
        if cur.type == type_name:
            return True
        cur = cur.parent
    return False


def _find_function_declarator(func_node):
    """Return the callable's own declarator WITHOUT descending into the body.

    Handles the wrapper cases: `Foo& Bar::baz()` -> reference_declarator ->
    function_declarator, and `Foo* baz()` -> pointer_declarator.
    """
    for c in func_node.children:
        if c.type == "function_declarator":
            return c
        if c.type in ("reference_declarator", "pointer_declarator"):
            for d in c.children:
                if d.type == "function_declarator":
                    return d
    return None


def _field_name(src: bytes, node) -> str | None:
    """Declared name of a field/variable declarator node (AST-based, avoids the
    regex fallback that picks up type keywords or initialiser values)."""
    if node is None:
        return None
    t = node.type
    if t in ("field_identifier", "identifier"):
        return _text(src, node)
    if t in ("pointer_declarator", "reference_declarator", "array_declarator",
             "parenthesized_declarator", "init_declarator"):
        d = _child_by_field(node, "declarator")
        r = _field_name(src, d) if d is not None else None
        if r:
            return r
        for c in node.children:
            r = _field_name(src, c)
            if r:
                return r
        return None
    d = _find_descendant(node, "field_identifier")
    return _text(src, d) if d is not None else None


def _split_declaration(src: bytes, node):
    """Return (type_text, [identifier names]) for a field/variable declaration."""
    if node.type == "field_declaration":
        left = []
        for c in node.children:
            if c.type in ("field_identifier", "init_declarator", ",", ";", "=", "default" ):
                continue
        # type text = up to first field_identifier
        text = _text(src, node)
        m = re.match(r"^(.*?)[\s&*]+([A-Za-z_]\w*)\s*(=|;|$)", text, re.S)
        names = re.findall(r"[A-Za-z_]\w*(?=\s*(?:=|,|;|\[|$))", text)
        # crude: the last identifier before = or ; is the name
        type_txt = text
        if m:
            type_txt = m.group(1).strip()
        return type_txt, ([names[-1]] if names else [])
    return _text(src, node), []


def _declarator_name(src: bytes, fd) -> str:
    d = _child_by_field(fd, "declarator")
    if d is None:
        return ""
    txt = _text(src, d)
    return txt.split("::")[-1].strip()


def _declarator_qualified_name(src: bytes, fd) -> str:
    d = _child_by_field(fd, "declarator")
    if d is None:
        return ""
    return _text(src, d)


def _split_qualified(qname: str):
    if "::" in qname:
        scope, name = qname.rsplit("::", 1)
        return scope, name
    return None, qname


def _declarator_return_type(src: bytes, func_node, fd) -> str | None:
    # Everything before the declarator, minus modifiers, is the return type
    text = _text(src, func_node)
    # return type is the source between start of node and start of function_declarator/
    # reference_declarator wrapper
    top = _top_declarator(func_node, fd)
    pre = src[func_node.start_byte:top.start_byte].decode("utf-8", "replace").strip()
    pre = re.sub(r"\b(virtual|static|inline|explicit|constexpr|friend)\b", "", pre).strip()
    return pre or None


def _top_declarator(func_node, fd):
    # if the function_declarator is wrapped (reference_declarator / pointer_declarator),
    # the return type is before the wrapper
    for c in func_node.children:
        if c.type in ("reference_declarator", "pointer_declarator") and _contains(c, fd):
            return c
    return fd


def _contains(ancestor, node) -> bool:
    return ancestor.start_byte <= node.start_byte and ancestor.end_byte >= node.end_byte


def _parse_params(src: bytes, fd) -> list[dict]:
    plist = _child_by_field(fd, "parameters")
    params = []
    if plist is None:
        return params
    for c in plist.children:
        if c.type == "parameter_declaration":
            txt = _text(src, c)
            name = ""
            # find the declared name: last identifier that's not a type keyword
            ids = [d for d in _walk(c) if d.type in ("identifier", "field_identifier")]
            # declared name is typically the last identifier in a parameter
            if ids:
                name = _text(src, ids[-1])
            # type = text minus the trailing name
            typ = txt
            if name and txt.rstrip().endswith(name):
                typ = txt[:txt.rfind(name)].strip()
            typ = re.sub(r"\s+", " ", typ)
            params.append({"name": name, "type": typ or txt, "code": txt})
    return params


def _strip_template_args(text: str) -> str:
    """`QVector<QString>::size` -> `QVector::size` (depth-aware).

    Operator names keep their punctuation: `operator<<` must not become
    `operator`, or every stream operator would collapse into one symbol.
    """
    if "operator" in text:
        return text
    out, depth = [], 0
    for ch in text:
        if ch == "<":
            depth += 1
            continue
        if ch == ">":
            depth = max(0, depth - 1)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


_BUILTIN_TYPES = {
    "void", "bool", "char", "short", "int", "long", "float", "double",
    "size_t", "ssize_t", "wchar_t", "char8_t", "char16_t", "char32_t", "auto",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "qint8", "qint16", "qint32", "qint64",
    "quint8", "quint16", "quint32", "quint64", "uchar", "ushort", "uint", "ulong",
}

_TYPE_NOISE = re.compile(
    r"\b(const|volatile|static|mutable|inline|extern|register|typename|struct|"
    r"class|enum|unsigned|signed|long|short)\b")


def _normalize_type(type_text: str) -> str:
    """Declared type -> the class name a method call should resolve against.

    `const QString &` -> `QString`; `std::shared_ptr<Foo>` -> `shared_ptr`
    (its methods are not Foo's, so resolving into Foo would be a lie).
    Built-in types return "" so they never become a lookup key.
    """
    if not type_text:
        return ""
    t = _strip_template_args(type_text)
    t = _TYPE_NOISE.sub(" ", t)
    t = t.replace("*", " ").replace("&", " ")
    parts = [p for p in re.split(r"[\s:]+", t) if p]
    if not parts or parts[-1] in _BUILTIN_TYPES:
        return ""
    if parts[0].startswith("std") or "std" in t.split("::"):
        # keep the wrapper type, not its template argument
        parts = [p for p in parts if p != "std"]
        return parts[0] if parts else ""
    return parts[-1]


def _callee_text(src: bytes, fn_node) -> str:
    """Text of a callee, **keeping its receiver**.

    `res.append(x)` -> `res.append`. Returning the bare `append` is what let a
    QString method call resolve to an unrelated `TextForm::append`, and made the
    caller's `uses` list name a function it never calls.
    """
    t = fn_node.type
    if t in ("identifier", "qualified_identifier", "field_identifier",
             "template_function", "destructor_name", "operator_name"):
        return _text(src, fn_node)
    if t == "field_expression":
        return _text(src, fn_node)          # receiver + "->"/"." + field
    if t == "template_function":
        return _text(src, fn_node)
    # fallback
    return _text(src, fn_node)


def _callable_kind(name: str, cls_name: str) -> str:
    if name == cls_name:
        return "Constructor"
    if name.startswith("~"):
        return "Destructor"
    return "Method"


def _chunk_with_context(onto: "CppOntology", e: Entity) -> str:
    fnote = f" [{e.file}]" if e.file else ""
    header = f"// {e.kind} {e.qname}{fnote} (lines {e.line_start}-{e.line_end})"
    return header + "\n" + e.code


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

CPP_EXTS = (".cpp", ".cc", ".cxx", ".c++", ".h", ".hpp", ".hh", ".hxx")


def build_from_file(path: str) -> CppOntology:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    onto = CppOntology(filename=path.replace("\\", "/").split("/")[-1])
    onto.build(text)
    onto.build_retriever()
    return onto


def build_from_files(paths: list[str], root: str | None = None,
                     with_retriever: bool = True) -> CppOntology:
    """Build one ontology over several C/C++ source files (multi-file mode).

    Pass `paths` relative to `root` (with `root` set) so entity ids and chunk
    provenance are stable and path-independent. A class declared in a `.h` and
    defined in the `.cpp` beside it share one entity (method-to-class and call
    edges resolve across files); same-named classes in different directories
    stay distinct (`class:config/Config` vs `class:server_app/config/Config`).
    """
    onto = CppOntology()
    texts = []
    for p in paths:
        rel = p.replace("\\", "/")
        full = os.path.join(root, rel) if root else p
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            texts.append((rel, f.read()))
    for _rel, text in texts:      # pass 1: class / namespace names
        onto.prescan(text)
    for rel, text in texts:       # pass 2: full extraction
        onto.add_file(text, rel)
    return onto.finish(with_retriever=with_retriever)


def build_from_dir(root: str, exts=CPP_EXTS, with_retriever: bool = True,
                   exclude=(), skip_names=()) -> CppOntology:
    """Recursively collect C/C++ files under `root` and build one ontology.

    Files are namespaced by their directory relative to `root`, so the ids do
    not depend on where the project happens to live on disk.

    `exclude` (directory names) and `skip_names` (file-name prefixes) keep
    generated trees out. They exist so a caller can hand the *same* filter used
    for freshness hashing — otherwise the graph and the hashes disagree about
    which files exist.
    """
    exset = set(exclude or ())
    paths = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in exset and not d.startswith("."))
        for fn in sorted(files):
            if not fn.lower().endswith(exts):
                continue
            if skip_names and fn.lower().startswith(tuple(skip_names)):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            paths.append(rel.replace("\\", "/"))
    return build_from_files(paths, root=root, with_retriever=with_retriever)


if __name__ == "__main__":
    import sys
    onto = build_from_file(sys.argv[1] if len(sys.argv) > 1 else "cpp_test_code_from_practice.txt")
    print(f"Entities: {len(onto.entities)}  Relations: {len(onto.relations)}  Chunks: {len(onto.chunks)}")
    for e in onto.entities.values():
        print(f"  {e.kind:12} {e.qname or e.name}")
