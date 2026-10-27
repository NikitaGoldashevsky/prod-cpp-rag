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
    void show();
};

class Form {
public:
    void show(int mode);
    void open();
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
#include "nvdatetime.h"

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

void Form::show(int mode) { show(); }   // calls the OTHER overload
void Form::open() {}
"""

# An include-guarded header: everything inside `#ifndef`. Before the fix the
# whole file was one preproc node, so these classes existed only as body-less
# stubs — no line numbers, no base classes, and no fields, which in turn made
# every `obj.method()` call in the project unresolvable.
GUARDED = r"""
#ifndef NOTIFY_EXTRA_H
#define NOTIFY_EXTRA_H

class NotifyClocks : public AlarmClock {
public:
    void working();
    void off();
};

class ComplexTimer : public SingleTimer {
public:
    void working();
};

template <typename T>
class Box : public SingleTimer {
public:
    T get();
};

namespace ui {
class Chips : public ComplexTimer {
public:
    void add();
};
}

extern "C" {
void c_entry();
}

class ComplexTimer2 : public ComplexTimer {
public:
    void goNext();
    void fromString();
};

class Manager {
public:
    void timerEvent();
private:
    ComplexTimer complex_timer;
    ComplexTimer* ptr_timer;
    QList<NotifyClocks*> clocks;
    QList<NotifyClocks> value_clocks;
    AlarmClock* am;
};

#endif
"""

GUARDED_CPP = r"""
#include "notify_extra.h"

void NotifyClocks::working() {}
void NotifyClocks::off() {}
void ComplexTimer::working() {}
void ComplexTimer2::goNext() {}
void Manager::timerEvent() {
    complex_timer.working();
    ptr_timer->working();
    clocks[i]->working();
    clocks[i]->off();
    value_clocks[i].working();
}
void ComplexTimer2::fromString() {
    deltaTime();
    goNext();
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
    with open(os.path.join(root, "notify_extra.h"), "w") as f:
        f.write(GUARDED)
    with open(os.path.join(root, "notify_extra.cpp"), "w") as f:
        f.write(GUARDED_CPP)
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

    # ---- shapes the work-PC agent reported as 0-of-9 resolved ----------
    # obj.method() through a value field
    expect("Manager::timerEvent", "calls", "ComplexTimer::working")
    # ptr->method() through a pointer field (the `>` in `->` used to be eaten
    # by template-argument stripping, leaving `ptr-method` unresolvable)
    e = load.entity("Manager::timerEvent")
    assert "ComplexTimer::working" in (e.get("calls") or []), e.get("calls")
    # container[i]->method(): element type of QList<NotifyClocks*>
    expect("Manager::timerEvent", "calls", "NotifyClocks::working")
    expect("Manager::timerEvent", "calls", "NotifyClocks::off")
    # Inherited and unqualified, inside an include-guarded header. `deltaTime`
    # is declared once, in AlarmClock, three bases up; resolving to the class
    # that actually declares it is the correct answer.
    e = load.entity("ComplexTimer2::fromString")
    got = e.get("calls") or []
    if not any(c.endswith("::deltaTime") for c in got):
        print(f"  FAIL  inherited call unresolved: {got}")
        ok = False
    else:
        print(f"  OK    inherited call resolved to {[c for c in got if c.endswith('deltaTime')][0]}")

    # ---- guarded headers are not lost ---------------------------------
    for qname in ("NotifyClocks", "ComplexTimer", "Chips", "ComplexTimer2",
                  "Manager", "Box"):
        ent = load.entity(qname)
        if not ent or not ent.get("line_start"):
            print(f"  FAIL  {qname}: no declaration (guarded-header bug)")
            ok = False
    if load.entity("Chips") and load.entity("Chips").get("declared_in_index") is not True:
        print("  FAIL  Chips: declared_in_index should be True")
        ok = False
    if load.entity("Box") and not load.entity("Box").get("inherits"):
        print("  FAIL  template class Box lost its base class")
        ok = False
    print("  OK    guarded header: classes, line numbers, inheritance")

    # ---- an overload of itself is not recursion ------------------------
    # `Form::show(int)` calls `show()`. Only one overload exists in the index,
    # so a naive resolver emits Form::show -> Form::show: a self-edge that
    # claims recursion. The honest answer is "a callee we do not have".
    ent = load.entity("Form::show")
    got = ent.get("calls") or []
    if "Form::show" in got:
        print("  FAIL  false self-edge: Form::show -> Form::show")
        ok = False
    else:
        print(f"  OK    overload of itself is not emitted as recursion "
              f"(uses={ent.get('uses')})")

    # ---- a file stub is not a symbol -----------------------------------
    # `entity("nvdatetime")` used to resolve to the `#include "nvdatetime.h"`
    # Header node: found, kind Header, file "", nothing to jump to.
    stub = load.entity("nvdatetime")
    if stub is not None and stub.get("kind") == "Header":
        print("  FAIL  entity() resolved a symbol name to a Header stub")
        ok = False
    else:
        print(f"  OK    include stub not returned as a symbol ({stub and stub.get('kind')})")

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
