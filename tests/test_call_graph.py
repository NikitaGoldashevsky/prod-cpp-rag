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

// One function, two call sites, only one of them guarded. "This function has a
// try/catch" does not answer "is THIS call guarded".
void guardedDelta() {}
void looseDelta() {}
void mixed() {
    try { guardedDelta(); } catch (...) {}
    looseDelta();
}
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

# A Qt form header: Q_OBJECT, Q_PROPERTY, public slots:, signals:, and a base
# class that lives outside the indexed tree. Before macro blanking, the parser
# produced ERROR nodes here and lost the member that followed each macro.
QT_HEADER = r"""
#ifndef QTFORM_H
#define QTFORM_H

class QtForm : public NvOutOfTreeBase
{
    Q_OBJECT
    Q_PROPERTY(int value READ value)

public:
    explicit QtForm(QWidget* parent = nullptr);
    void run();
    void working();

public slots:
    void hide(int mode);
    void settle();

signals:
    void changed();

protected:
    void keyPressEvent(QKeyEvent* e);

private:
    int value_;
};

class Widget0 { public: void run(); void working(); };
class Widget1 { public: void run(); void working(); };

class QtMgr
{
public:
    void timerEvent();
    void working();
private:
    void onTick();
};

#endif
"""

QT_SOURCE = r"""
#include "qtform.h"

QtForm::QtForm(QWidget* parent) {}
void QtForm::run() { working(); qtHelper(); }
void QtForm::working() {}
void QtForm::hide(int mode) { settle(); }
void QtForm::settle() {}
void QtForm::keyPressEvent(QKeyEvent* e) { base::keyPressEvent(e); }

void Widget0::run() {}
void Widget0::working() {}
void Widget1::run() {}
void Widget1::working() {}

void QtMgr::working() {}
void QtMgr::timerEvent() {
    QList<Widget0*> forms = findChildren<Widget0*>();
    forms[i]->run();
    forms[i]->working();
    working();
}
void QtMgr::onTick() { hide(2); }

void qtHelper() {}
"""

# Virtual dispatch, a file-scope object, a forward declaration, and the shapes
# an audit flagged as mislabelled in `uses`.
VIRT_HEADER = r"""
#ifndef VIRT_H
#define VIRT_H

class NvDateTime;                 // forward declaration only

class BaseTimer
{
public:
    virtual int deltaTime();
    static void helper();
};

class VirtTimer : public BaseTimer
{
public:
    int deltaTime();              // override, `virtual` not repeated
};

class VirtMgr
{
public:
    void timerEvent();
    void formatting();
};

#endif
"""

VIRT_SOURCE = r"""
#include "virt.h"

static BaseTimer base_timer;      // file-scope object of an in-tree type

int BaseTimer::deltaTime() { return 0; }
void BaseTimer::helper() {}
int VirtTimer::deltaTime() { return 1; }

void VirtMgr::timerEvent()
{
    base_timer.deltaTime();
    BaseTimer* p = new VirtTimer();
    p->deltaTime();
}

void VirtMgr::formatting()
{
    QString("%1:%2").arg(1).arg(2);
    NvDistanceFormat(3);
}
"""

CHAIN_SRC = r"""
class ChainBase { public: virtual void off(); };
class ChainMid : public ChainBase { public: void off(); };
class ChainLeaf : public ChainMid { public: void off(); };
class ChainOut : public NvExternalBase { public: void g(); };

class ComplexTimerForm : public NvAutoForm, public NvStatic<ComplexTimerForm>
{
public:
    void show();
};

class TrackForm
{
public:
    void setup();
    void onSaved();
};

void ChainBase::off() {}
void ChainMid::off() {}
void ChainLeaf::off() {}
void ComplexTimerForm::show() {}
void TrackForm::setup()
{
    connect(saved, SIGNAL(pressed()), this, SLOT(onSaved()));
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
    with open(os.path.join(root, "virt.h"), "w") as f:
        f.write(VIRT_HEADER)
    with open(os.path.join(root, "virt.cpp"), "w") as f:
        f.write(VIRT_SOURCE)
    with open(os.path.join(root, "qtform.h"), "w") as f:
        f.write(QT_HEADER)
    with open(os.path.join(root, "qtform.cpp"), "w") as f:
        f.write(QT_SOURCE)
    with open(os.path.join(root, "notify_extra.h"), "w") as f:
        f.write(GUARDED)
    with open(os.path.join(root, "notify_extra.cpp"), "w") as f:
        f.write(GUARDED_CPP)
    os.environ["PROD_RAG_DATA"] = os.path.join(tmp, "data")
    idx_mod.build("cg", root, verbose=False)
    return idx_mod.load("cg").onto


def _has_use(onto, label: str) -> bool:
    return any(r.p == "callsExternal" and r.o == label for r in onto.relations)


SPLIT_HDR = r"""
#ifndef SPLIT_H
#define SPLIT_H

class SplitBase
{
public:
    virtual void run();
    virtual void tick();
};

class SplitDer : public SplitBase
{
public:
    void run() override;
    class Inner : public SplitBase
    {
    public:
        void tick() override;
    };
};

class SplitMgr
{
public:
    void go();
};

#endif
"""

SPLIT_SRC = r"""
#include "shared/split.h"

void SplitBase::run() {}
void SplitBase::tick() {}
void SplitDer::run() { tick(); }
void SplitDer::Inner::tick() {}
void SplitMgr::go()
{
    SplitDer d;
    d.run();
    SplitDer::Inner in;
    in.tick();
}
"""


def check_header_source_split(tmp: str) -> bool:
    """A header in one directory, its source in another, joined by #include.

    Entity ids are namespaced by directory so same-named classes in different
    folders stay distinct. That split every realistic layout in two: the class
    existed once from the header (with its base) and once from the .cpp (a
    body-less stub), so inheritance, overrides and inherited calls were all
    computed on two halves that each looked incomplete.
    """
    root = os.path.join(tmp, "split")
    os.makedirs(os.path.join(root, "src", "shared"))
    with open(os.path.join(root, "src", "shared", "split.h"), "w") as f:
        f.write(SPLIT_HDR)
    with open(os.path.join(root, "src", "split.cpp"), "w") as f:
        f.write(SPLIT_SRC)
    idx_mod.build("split", root, verbose=False)
    onto = idx_mod.load("split").onto
    from prod_rag.store import MemoryStore as _MS
    st = _MS(onto)
    ok = True

    classes = [e.qname for e in onto.entities.values() if e.kind == "Class"]
    for name in ("SplitBase", "SplitDer", "SplitDer::Inner", "SplitMgr"):
        if classes.count(name) != 1:
            print(f"  FAIL  {name}: {classes.count(name)} entities after the "
                  f"header/source merge")
            ok = False
    if not ok:
        return False
    print("  OK    header and source merged across directories")

    der = st.entity("SplitDer")
    if der.get("inherits_resolved") != ["SplitBase"]:
        print(f"  FAIL  SplitDer base unresolved: {der.get('inherits_resolved')}")
        ok = False
    # inherited unqualified call inside the .cpp half
    if "SplitBase::tick" not in (st.entity("SplitDer::run") or {}).get("calls", []):
        print(f"  FAIL  inherited call lost after merge: "
              f"{(st.entity('SplitDer::run') or {}).get('calls')}")
        ok = False
    # qualified-type local, then a call through it
    if "SplitDer::Inner::tick" not in (st.entity("SplitMgr::go") or {}).get("calls", []):
        print(f"  FAIL  qualified local type not resolved: "
              f"{(st.entity('SplitMgr::go') or {}).get('calls')}")
        ok = False
    # a nested class must still be a class, and override
    inner = st.entity("SplitDer::Inner")
    inner_tick = st.entity("SplitDer::Inner::tick")
    if not inner_tick or inner_tick.get("overrides") != ["SplitBase::tick"]:
        print(f"  FAIL  nested class override: {inner_tick and inner_tick.get('overrides')}")
        ok = False
    if not inner or not inner.get("line_start"):
        print("  FAIL  nested class has no declaration line")
        ok = False
    # subclasses are visible from the base
    if st.entity("SplitBase").get("inherited_by") != ["SplitDer", "SplitDer::Inner"]:
        print(f"  FAIL  inherited_by: {st.entity('SplitBase').get('inherited_by')}")
        ok = False
    print("  OK    inherited call, nested override, and inherited_by all resolve")
    return ok


def check_overload_order(tmp: str) -> bool:
    """Same overloads, but the header sorts *before* the .cpp.

    This ordering used to delete an overload outright: the definition of
    `s(int)` collided with the id of `s()`, the twin merge kept the definition
    but wrote it under the colliding id, and `s()` ceased to exist.
    """
    root = os.path.join(tmp, "order")
    os.makedirs(root)
    hdr = ("#ifndef H\n#define H\nclass A { public: void s(); void s(int); };\n#endif\n")
    src = "void A::s() {}\nvoid A::s(int) {}\n"
    with open(os.path.join(root, "a.h"), "w") as f:
        f.write(hdr)
    with open(os.path.join(root, "b.cpp"), "w") as f:
        f.write(src)
    idx_mod.build("order", root, verbose=False)
    onto = idx_mod.load("order").onto
    sigs = sorted(tuple(p["type"] for p in e.params) for e in onto.entities.values()
                  if (e.qname or "") == "A::s")
    if sigs != [(), ("int",)]:
        print(f"  FAIL  overloads lost when the header sorts first: {sigs}")
        return False
    print("  OK    both overloads survive either file order")
    return True


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

    # ---- Qt macros must not eat members --------------------------------
    for qname in ("QtForm::hide", "QtForm::settle", "QtForm::changed",
                  "QtForm::keyPressEvent", "QtForm::value_", "QtForm::run"):
        if not load.entity(qname):
            print(f"  FAIL  {qname} missing: a Qt macro swallowed it")
            ok = False
    for junk in ("QtForm::Q_OBJECT", "QtForm::Q_PROPERTY"):
        if load.entity(junk):
            print(f"  FAIL  macro became an entity: {junk}")
            ok = False
    e = load.entity("QtForm::QtForm")
    if not e or not e.get("params"):
        print(f"  FAIL  constructor lost its default-argument parameter: {e and e.get('params')}")
        ok = False
    print("  OK    Q_OBJECT / Q_PROPERTY / slots / signals parse as members")

    # ---- container of pointers, declared as a local --------------------
    expect("QtMgr::timerEvent", "calls", "Widget0::run")
    expect("QtMgr::timerEvent", "calls", "Widget0::working")
    expect("QtMgr::timerEvent", "calls", "QtMgr::working")
    if "Widget0::run" in (load.entity("QtMgr::timerEvent") or {}).get("uses", []):
        print("  FAIL  container element type not resolved")
        ok = False

    # ---- free function with a body still resolves ----------------------
    expect("QtForm::run", "calls", "qtHelper")

    # ---- out-of-tree base: a bare name is not a header free function ---
    # `hide(2)` inside QtMgr (no such member, base chain unknown) must not
    # become a call to a `hide` declared in some other header.
    e = load.entity("QtMgr::onTick")
    if e and "hide" in (e.get("calls") or []):
        print(f"  FAIL  bare name bound to a header free function: {e.get('calls')}")
        ok = False
    # ... and `base::keyPressEvent(e)` must not become recursion
    e = load.entity("QtForm::keyPressEvent")
    if e and "QtForm::keyPressEvent" in (e.get("calls") or []):
        print("  FAIL  base::keyPressEvent bound to itself")
        ok = False

    # ---- a file-scope object has a type --------------------------------
    # `static BaseTimer base_timer;` spells the type in the same file; a call
    # through it was reported external with the receiver unknown.
    expect("VirtMgr::timerEvent", "calls", "BaseTimer::deltaTime")

    # ---- virtual dispatch is disclosed, not hidden ---------------------
    # The static call lands on BaseTimer::deltaTime; at runtime it may reach
    # NotifyClocks::deltaTime. Both facts must be visible.
    base = load.entity("BaseTimer::deltaTime")
    derived = load.entity("VirtTimer::deltaTime")
    if not base or base.get("overridden_by") != ["VirtTimer::deltaTime"]:
        print(f"  FAIL  base does not list its override: {base and base.get('overridden_by')}")
        ok = False
    if not derived or derived.get("overrides") != ["BaseTimer::deltaTime"]:
        print(f"  FAIL  override does not name its base: {derived and derived.get('overrides')}")
        ok = False
    if not base or "overridden by" not in (base.get("fact_card") or ""):
        print("  FAIL  fact card hides the override")
        ok = False
    print("  OK    virtual dispatch disclosed: base lists override, card says so")

    # ---- a forward declaration is not a definition ---------------------
    fwd = load.entity("NvDateTime")
    if fwd and fwd.get("is_definition") is not False:
        print(f"  FAIL  forward declaration claims to be a definition: {fwd.get('is_definition')}")
        ok = False
    print("  OK    forward declaration marked is_definition=False")

    # ---- `uses` labels say what kind of thing they are -----------------
    # Reported from an audit: type names looked like calls, a chained `.arg()`
    # lost its receiver, and a construction looked like a variable.
    for label, why in (("QString().arg", "chained call keeps its root"),
                       ("NvDistanceFormat()", "a construction says so")):
        if not _has_use(onto, label):
            print(f"  FAIL  uses label missing: {label} ({why})")
            ok = False
    print("  OK    uses labels distinguish constructions from members")

    # ---- third-generation override, and the point of the chain --------
    # `Leaf : Mid : Base`, only Base says `virtual`. Each link must be visible:
    # asking "what breaks if I change Base::off" has to name Leaf as well.
    chain = os.path.join(tmp, "chain")
    os.makedirs(chain)
    with open(os.path.join(chain, "chain.cpp"), "w") as fh:
        fh.write(CHAIN_SRC)
    idx_mod.build("chain", chain, verbose=False)
    from prod_rag.store import MemoryStore as _MS
    cs = _MS(idx_mod.load("chain").onto)
    base_off = cs.entity("ChainBase::off")
    leaf_off = cs.entity("ChainLeaf::off")
    if not base_off or sorted(base_off.get("overridden_by") or []) != [
            "ChainLeaf::off", "ChainMid::off"]:
        print(f"  FAIL  transitive overrides: base lists "
              f"{base_off and base_off.get('overridden_by')}")
        ok = False
    elif not leaf_off or "ChainBase::off" not in (leaf_off.get("overrides") or []):
        print(f"  FAIL  grandchild override not linked: {leaf_off and leaf_off.get('overrides')}")
        ok = False
    else:
        print("  OK    three-generation override chain, both directions")

    # ---- an out-of-tree base is disclosed, not silently unresolved ----
    if not base_off or base_off.get("inherits_unknown"):
        pass
    out_cls = cs.entity("ChainOut")
    if out_cls and out_cls.get("inherits_unknown") != ["NvExternalBase"]:
        print(f"  FAIL  out-of-tree base not disclosed: "
              f"{out_cls.get('inherits_resolved')} / {out_cls.get('inherits_unknown')}")
        ok = False
    else:
        print("  OK    out-of-tree base disclosed in the payload")

    # ---- a class is never its own base --------------------------------
    tpl = load.entity("ComplexTimerForm")
    if tpl and "ComplexTimerForm" in (tpl.get("inherits") or []):
        print(f"  FAIL  template argument recorded as a base: {tpl.get('inherits')}")
        ok = False
    else:
        print("  OK    template argument is not a base class")

    # ---- SIGNAL()/SLOT() contents are not calls -----------------------
    setup = load.entity("TrackForm::setup")
    if setup and any("onSaved" in c for c in (setup.get("calls") or [])):
        print(f"  FAIL  signal wiring recorded as a call: {setup.get('calls')}")
        ok = False
    else:
        print("  OK    SIGNAL()/SLOT() contents are not calls")

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

    # ---- per-call guardedness -------------------------------------------
    # The canonical question "who calls X and is it guarded" was answerable
    # only from `has_try_catch`, which is a property of the *function*.
    mixed = load.entity("mixed")
    if not mixed:
        print("  FAIL  mixed(): entity missing")
        ok = False
    else:
        loose = mixed.get("unguarded_calls") or []
        if loose != ["looseDelta"]:
            print(f"  FAIL  unguarded_calls = {loose}, expected ['looseDelta']")
            ok = False
        else:
            print("  OK    per-call guardedness: guardedDelta guarded, "
                  "looseDelta not")
        card = mixed.get("fact_card") or ""
        if "NOT inside a try block" not in card:
            print(f"  FAIL  fact card does not say which call is unguarded: {card}")
            ok = False

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

    if not check_header_source_split(tmp):
        ok = False
    if not check_overload_order(tmp):
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
