# -*- coding: utf-8 -*-
"""Smoke tests: no network, no model, no external server.

    python -m pytest tests/ -q        (or)      python tests/test_smoke.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prod_rag import index as idx_mod          # noqa: E402
from prod_rag.service import Service           # noqa: E402

CPP_MAIN = r"""
#include <stdexcept>
#include "widget.h"

namespace shop {

Widget::Widget(int id, const std::string& name) : id_(id), name_(name) {
    if (id < 0) {
        throw std::invalid_argument("negative id");
    }
}

int Widget::getId() const { return id_; }

void Widget::rename(const std::string& name) {
    name_ = name;
    notify();
}

void Widget::notify() {
    try {
        broadcast(name_);
    } catch (const std::exception& e) {
        // swallow
    }
}

void Widget::broadcast(const std::string& msg) {
    render(msg);
}

}  // namespace shop
"""

CPP_RANK = r"""
class TcpClient {
public:
    void connect();
};
void TcpClient::connect() {}

struct Machine {
    void resetCpu();
    void adjustCpu();
};
void Machine::resetCpu() {}
void Machine::adjustCpu() {}
"""

CPP_HDR = r"""
#pragma once
#include <string>

namespace shop {

class Widget {
public:
    Widget(int id, const std::string& name);
    int getId() const;
    void rename(const std::string& name);
    void notify();
    void broadcast(const std::string& msg);
    void render(const std::string& msg);
private:
    int id_;
    std::string name_;
};

}  // namespace shop
"""

CPP_RENDER = r"""
#include "widget.h"

namespace shop {
void Widget::render(const std::string& msg) {
    (void)msg;
}
}
"""


def make_tree(tmp: str) -> str:
    root = os.path.join(tmp, "proj")
    os.makedirs(os.path.join(root, "src"), exist_ok=True)
    with open(os.path.join(root, "src", "widget.cpp"), "w") as f:
        f.write(CPP_MAIN)
    with open(os.path.join(root, "src", "widget.h"), "w") as f:
        f.write(CPP_HDR)
    with open(os.path.join(root, "src", "render.cpp"), "w") as f:
        f.write(CPP_RENDER)
    # noise that must be skipped
    os.makedirs(os.path.join(root, "build"), exist_ok=True)
    with open(os.path.join(root, "build", "generated.cpp"), "w") as f:
        f.write("int generated() { return 0; }\n")
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write("# not code\n")
    return root


def run(tmp: str):
    os.environ["PROD_RAG_DATA"] = os.path.join(tmp, "data")
    root = make_tree(tmp)

    # ---- build -----------------------------------------------------------
    meta = idx_mod.build("proj", root, verbose=False)
    s = meta["stats"]
    assert s["files"] == 3, s          # build/ excluded, README not C++
    assert s["entities"] > 10, s
    assert s["chunks"] >= 3, s
    print(f"  [OK] build: {s['files']} files, {s['entities']} entities, {s['chunks']} chunks")

    # ---- load + search ---------------------------------------------------
    svc = Service(["proj"])
    r = svc.search("what throws when the id is negative", top_k=4)
    assert r["chunks"], "no chunks returned"
    names = [c["qname"] for c in r["chunks"]]
    assert any("Widget" in n for n in names), names
    print(f"  [OK] search returns chunks: {names[:3]}")

    # ---- entity ----------------------------------------------------------
    e = svc.entity("Widget::rename")
    assert e["found"], e
    assert e["qname"] == "Widget::rename", e["qname"]
    assert "Widget::notify" in e["calls"], e["calls"]
    assert e["file"].endswith("widget.cpp"), e["file"]
    print(f"  [OK] entity: {e['qname']} calls {e['calls']}")

    # ---- exceptions are facts, not strings -------------------------------
    notify = svc.entity("Widget::notify")
    assert notify["has_try_catch"] is True, notify
    ctor = svc.entity("Widget::Widget")
    assert ctor["has_try_catch"] is False, ctor
    assert any(t["type"] == "std::invalid_argument" for t in ctor["throw_sites"]), ctor
    print(f"  [OK] control flow: notify has try/catch, ctor throws "
          f"{[t['type'] for t in ctor['throw_sites']]}")

    # ---- neighbours ------------------------------------------------------
    n = svc.neighbours("Widget::notify", hops=1)
    assert n["found"] and len(n["entities"]) > 1, n
    print(f"  [OK] neighbourhood: {len(n['entities'])} entities")

    # ---- find ------------------------------------------------------------
    f = svc.find("Widget*")
    assert f["count"] >= 1, f
    assert all(h["match"] == "glob" for h in f["matches"]), f
    print(f"  [OK] find: {f['count']} matches (glob)")

    # ---- freshness: fresh, then stale, then reindex ----------------------
    idx = svc.get("proj")
    fresh = idx.freshness()
    assert fresh["state"] == "fresh", fresh
    assert all(d["status"] == "fresh" for d in fresh["folders"]), fresh["folders"]

    with open(os.path.join(root, "src", "render.cpp"), "a") as fh:
        fh.write("\n// a comment that changes the file\n")
    stale = idx.freshness(rescan=True)
    assert stale["state"] == "stale", stale
    assert stale["changed"] == ["src/render.cpp"], stale["changed"]
    by_path = {d["path"]: d for d in stale["folders"]}
    assert by_path["src"]["status"] == "stale", by_path["src"]
    assert by_path["."]["status"] == "stale", "root subtree hash must see the change"
    print(f"  [OK] freshness: stale after edit ({stale['detail']}), "
          f"root marked stale too")

    svc.reindex("proj")
    assert svc.get("proj").freshness()["state"] == "fresh"
    print("  [OK] reindex restores fresh")

    # ---- removed + added files ------------------------------------------
    os.remove(os.path.join(root, "src", "render.cpp"))
    with open(os.path.join(root, "src", "extra.cpp"), "w") as fh:
        fh.write("namespace shop { void extra() {} }\n")
    st = svc.get("proj").freshness(rescan=True)
    assert "src/render.cpp" in st["removed"], st["removed"]
    assert "src/extra.cpp" in st["added"], st["added"]
    print("  [OK] add/remove detection")

    # ---- CLI -------------------------------------------------------------
    env = dict(os.environ)
    cli = [sys.executable, "-m", "prod_rag", "index", "list"]
    out = subprocess.run(cli, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "proj" in out.stdout, out.stdout
    st = subprocess.run([sys.executable, "-m", "prod_rag", "index", "status", "proj"],
                        capture_output=True, text=True, env=env)
    assert st.returncode in (0, 2), (st.returncode, st.stderr)
    print("  [OK] CLI list/status")

    # ---- find ranks word matches above accidents -------------------------
    # "Tcp" is really a substring of "resetCpu" (rese-tcp-u), and substring
    # search is correct — but a real TcpClient must sort first, and every hit
    # must say which kind of match it is. Otherwise an agent believes a symbol
    # exists when it only matched mid-word.
    rank_root = os.path.join(tmp, "rank")
    os.makedirs(rank_root)
    with open(os.path.join(rank_root, "cpu.cpp"), "w") as fh:
        fh.write(CPP_RANK)
    idx_mod.build("rank", rank_root, verbose=False)
    svc3 = Service(["rank"])
    hits = svc3.find("Tcp")["matches"]
    assert hits and hits[0]["qname"].startswith("Tcp"), hits
    assert hits[0]["match"] in ("exact", "prefix", "word"), hits[0]
    assert any(h["match"] == "substring" for h in hits), hits
    print(f"  [OK] find ranks {hits[0]['qname']} ({hits[0]['match']}) above "
          f"substring noise ({len(hits)} hits)")

    # ---- generated code stays out of the index --------------------------
    # A moc_ file defines the same class without a body; indexing it makes a
    # class name resolve to a stub with no members, calls or callers.
    gen_root = os.path.join(tmp, "gen")
    os.makedirs(os.path.join(gen_root, "obj"))
    with open(os.path.join(gen_root, "real.cpp"), "w") as fh:
        fh.write("class Widget { public: void run(); };\nvoid Widget::run() {}\n")
    for path in ("obj/moc_widget.cpp", "moc_widget2.cpp", "ui_widget.h"):
        with open(os.path.join(gen_root, path), "w") as fh:
            fh.write("class Widget { public: void run(); };\n")
    meta_g = idx_mod.build("gen", gen_root, verbose=False)
    assert meta_g["stats"]["files"] == 1, meta_g["stats"]
    print("  [OK] generated moc_/ui_/obj files are not indexed")

    # ---- multi-index facade ---------------------------------------------
    idx_mod.build("proj2", root, verbose=False)
    svc2 = Service(None)
    res = svc2.search("Widget", top_k=2)
    assert "indexes" in res and len(res["indexes"]) >= 2, list(res)
    print(f"  [OK] multi-index search returns {len(res['indexes'])} per-index groups")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="prodrag-test-")
    try:
        print("PROD_RAG smoke tests")
        run(tmp)
        print("  ALL PASSED")
        return 0
    except AssertionError as e:
        print(f"  FAILED: {e}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
