#!/usr/bin/env bash
# neo4j_setup.sh — install a local Neo4j Community server with no root and no Docker.
#
# Neo4j's own download site is geo-blocked in some networks, but the official
# image is published to Docker Hub, whose registry accepts plain HTTPS. This
# script pulls the image layers with curl, verifies every layer digest, and
# unpacks them into ~/.local/opt/neo4j/rootfs. The image already contains a JDK,
# so nothing else has to be installed.
#
#   ./neo4j_setup.sh                 # install (idempotent) and print next steps
#   NEO4J_TAG=2026.10.0-community ./neo4j_setup.sh
#
# Afterwards:
#   ~/.local/bin/neo4j-local start   # http://127.0.0.1:7474, bolt://127.0.0.1:7687
#   ~/.local/bin/neo4j-local stop
#
# Credentials are generated once and written to
# ~/.local/opt/neo4j/credentials.env (mode 600, outside this repository).
set -euo pipefail

TAG="${NEO4J_TAG:-2026.10.0-community}"
IMAGE="library/neo4j"
OPT="$HOME/.local/opt/neo4j"
ROOT="$OPT/rootfs"
NEO4J_HOME="$ROOT/var/lib/neo4j"
LAYERS="$OPT/layers"
JDK="$ROOT/opt/java/openjdk"

echo "==> Neo4j image $IMAGE:$TAG -> $ROOT"
mkdir -p "$LAYERS"

python3 - "$TAG" "$LAYERS" <<'PY'
import hashlib, json, os, subprocess, sys, urllib.request

tag, layers_dir = sys.argv[1], sys.argv[2]
REPO = "library/neo4j"
REG = "https://registry-1.docker.io/v2"

def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()

token = json.loads(get(
    "https://auth.docker.io/token?service=registry.docker.io&scope=repository:%s:pull" % REPO
))["token"]
auth = {"Authorization": "Bearer " + token,
        "Accept": "application/vnd.oci.image.index.v1+json,"
                  "application/vnd.docker.distribution.manifest.list.v2+json,"
                  "application/vnd.oci.image.manifest.v1+json"}

index = json.loads(get(f"{REG}/{REPO}/manifests/{tag}", auth))
if "manifests" in index:                       # multi-arch index -> pick amd64
    amd = [m for m in index["manifests"]
           if m.get("platform", {}).get("architecture") == "amd64"
           and m.get("platform", {}).get("os") == "linux"
           and m.get("platform", {}).get("variant") != "v8"]
    digest = amd[0]["digest"]
    manifest = json.loads(get(f"{REG}/{REPO}/manifests/{digest}", auth))
else:
    manifest = index

for i, layer in enumerate(manifest["layers"]):
    d = layer["digest"]
    fn = os.path.join(layers_dir, f"{i:02d}_{d.split(':')[1][:16]}.tar.gz")
    if not (os.path.exists(fn) and os.path.getsize(fn) == layer["size"]):
        print(f"    downloading layer {i + 1}/{len(manifest['layers'])} "
              f"({layer['size'] // 1024 // 1024} MiB)", flush=True)
        data = get(f"{REG}/{REPO}/blobs/{d}", auth)
        assert hashlib.sha256(data).hexdigest() == d.split(":")[1], "digest mismatch: " + d
        open(fn, "wb").write(data)
    else:
        print(f"    layer {i + 1} already present")
print("    all layers verified")
PY

echo "==> extracting"
mkdir -p "$ROOT"
# --skip-old-files keeps an existing database and our neo4j.conf additions intact
for f in "$LAYERS"/*.tar.gz; do tar -xzf "$f" -C "$ROOT" --skip-old-files; done

# The image wires data/ and logs/ to absolute container paths via symlinks.
for d in data logs; do
  if [ -L "$NEO4J_HOME/$d" ]; then unlink "$NEO4J_HOME/$d"; fi
  mkdir -p "$NEO4J_HOME/$d"
done
mkdir -p "$NEO4J_HOME/run" "$NEO4J_HOME/import" "$NEO4J_HOME/certificates"

if ! grep -q "local diploma setup" "$NEO4J_HOME/conf/neo4j.conf"; then
  cat >> "$NEO4J_HOME/conf/neo4j.conf" <<'EOF'

# --- local diploma setup (tarball extracted from the official image) ---
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1g
server.memory.pagecache.size=512m
server.default_listen_address=127.0.0.1
EOF
fi

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/neo4j-local" <<'EOF'
#!/bin/bash
# Local Neo4j Community (official image rootfs, no Docker).
export JAVA_HOME="$HOME/.local/opt/neo4j/rootfs/opt/java/openjdk"
export NEO4J_HOME="$HOME/.local/opt/neo4j/rootfs/var/lib/neo4j"
exec "$NEO4J_HOME/bin/neo4j" "$@"
EOF
chmod +x "$HOME/.local/bin/neo4j-local"

if [ ! -f "$OPT/credentials.env" ]; then
  PW=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
  JAVA_HOME="$JDK" "$NEO4J_HOME/bin/neo4j-admin" dbms set-initial-password "$PW" >/dev/null 2>&1
  printf 'NEO4J_URI=neo4j://127.0.0.1:7687\nNEO4J_USER=neo4j\nNEO4J_PASSWORD=%s\n' "$PW" \
    > "$OPT/credentials.env"
  chmod 600 "$OPT/credentials.env"
  echo "==> generated credentials in $OPT/credentials.env"
else
  echo "==> keeping existing $OPT/credentials.env"
fi

JAVA_HOME="$JDK" "$NEO4J_HOME/bin/neo4j" version 2>/dev/null | sed 's/^/    /'

cat <<EOF

Installed. Start it with:

    ~/.local/bin/neo4j-local start          # or 'console' to watch the log
    ~/.local/bin/neo4j-local status

Then load the ontology:

    .venv/bin/pip install neo4j
    .venv/bin/python neo4j_store.py --file cpp_test_code_from_practice.txt --reset --parity

Browser UI: http://127.0.0.1:7474  (user "neo4j", password in $OPT/credentials.env)
EOF
