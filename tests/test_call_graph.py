# -*- coding: utf-8 -*-
"""Call-graph resolution: who calls what must be a fact, not a name guess.

The measured failure this guards against (reported from a real 667-file
submodule): `SingleTimer::toLable` showed `calls=[]` although its body calls
`deltaTime()`, and `SingleTimer::toString` claimed to call `TextForm::append`
although its body calls `res.append(...)` on a `QString`. One false negative and
one false positive from the same rule: resolve a bare callee by "some entity
ends with that name".

    python tests/test_call_graph.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prod_rag import core as oc                # noqa: E402
from prod_rag import index as idx_mod          # noqa: E402
from prod_rag import store as store_mod        # noqa: E402

HEADER = r"""
#pragma once

class QString;

class AlarmClock {
public:
    qint64 deltaTime();
    virtual QString toString();
protected:
    QString label_;
};

class SingleTimer : public AlarmClock {
public:
    QString toLable();
    QString toString();
    void tick();
private:
    TextForm form_;
    int count_;
};

class TextForm {
public:
    void append(const QString& s);
    bool append(int n);
    void show();
};

class Widget {
public:
    void run();
    void render(const QString& s);
};

class Socket {
public:
    void connect();
    void openGlobal();
    void openMember();
};

void helper(int x);
"""

SOURCE = r"""
#include "notifyclocks.h"

qint64 AlarmClock::deltaTime() { return 5; }

QString SingleTimer::toLable() {
    qint64 t = deltaTime();
    return label_ + t;
}

QString SingleTimer::toString() {
    QString res = label_;
    res.append("x");
    form_.show();
    return res;
}

void SingleTimer::tick() {
    count_++;
    render("tick");
}

void Widget::run() {
    helper(3);
}

void Widget::render(const QString& s) {
}

void Socket::connect() {}

void Socket::openGlobal() {
    ::connect(1, 0, 0);        // libc, explicitly global scope
}

void Socket::openMember() {
    connect();                 // this->connect
}

void Remote::run() {
}
"""

TEXTFORM = r"""
class TextForm {
public:
    void append(const QString& s) {}
    bool append(int n) { return false; }
    void show() {}
};
"""


def build(tmp: str):
    root = os.path.join(tmp, "proj")
    os.makedirs(root)
    with open(os.path.join(root, "notifyclocks.h"), "w") as f:
        f.write(HEADER)
    with open(os.path.join(root, "notifyclocks.cpp"), "w") as f:
        f.write(SOURCE)
    with open(os.path.join(root, "textform.cpp"), "w") as f:
        f.write(TEXTFORM)
    os.environ["PROD_RAG_DATA"] = os.path.join(tmp, "data")
    idx_mod.build("cg", root, verbose=False)
    return idx_mod.load("cg").onto


def run(tmp: str) -> int:
    onto = build(tmp)
    load = store_mod.MemoryStore(onto)
    ok = True

    def expect(qname, key, wanted, exact=True):
        nonlocal ok
        e = load.entity(qname)
        if not e:
            print(f"  FAIL  {qname}: entity missing")
            ok = False
            return
        got = e.get(key) or []
        hit = wanted in got
        if hit != exact:
            print(f"  FAIL  {qname} {key}: {'missing' if exact else 'unexpected'} "
                  f"{wanted!r} in {got}")
            ok = False
        else:
            print(f"  OK    {qname} {key} {'has' if exact else 'does not have'} "
                  f"{wanted!r}")

    # inherited, unqualified call: the false negative
    expect("SingleTimer::toLable", "calls", "AlarmClock::deltaTime")
    # member call through a local variable of a known type: the false positive
    expect("SingleTimer::toString", "calls", "TextForm::append", exact=False)
    # ... and the same body's genuine member call through a field of known type
    expect("SingleTimer::toString", "calls", "TextForm::show")
    # a call to a method of another class must not be invented from a bare name
    expect("SingleTimer::tick", "calls", "Widget::render", exact=False)
    # free function in the same file
    expect("Widget::run", "calls", "helper")
    # `::connect` asks for global scope and must not become a self-edge
    expect("Socket::openGlobal", "calls", "Socket::connect", exact=False)
    # ... while an unqualified `connect()` inside a member does mean this->connect
    expect("Socket::openMember", "calls", "Socket::connect")

    # overloads: argument count picks the right one
    e = load.entity("SingleTimer::toString")
    top = [s for s in (e or {}).get("uses") or []]
    print(f"  ..    unresolved/external callees: {top}")
    e = load.entity("TextForm::append")
    print(f"  ..    overloads kept: {e and e['id']}")

    # every call edge must point at a real entity id
    for rel in onto.relations:
        if rel.p in ("calls", "references") and not rel.literal:
            if rel.o not in onto.entities:
                print(f"  FAIL  dangling {rel.p} edge {rel.s} -> {rel.o}")
                ok = False

    # a class declared outside the indexed tree must say so instead of
    # pretending line 0 is a location
    e = load.entity("SingleTimer")
    print(f"  ..    declared class: {e.get('file')}:{e.get('line_start')} "
          f"declared={e.get('declared_in_index')}")
    if e.get("declared_in_index") is not True:
        print("  FAIL  a declared class must report declared_in_index=True")
        ok = False
    e = load.entity("Remote")
    print(f"  ..    undeclared class: line={e.get('line_start')} "
          f"declared={e.get('declared_in_index')} "
          f"first_member={e.get('first_member')}")
    if e.get("declared_in_index") is not False or not e.get("first_member"):
        print("  FAIL  a class defined outside the tree must say so and point "
              "at a member definition")
        ok = False

    print("\nCALL GRAPH:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="prodrag-cg-")
    try:
        print("Call-graph resolution")
        return run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
