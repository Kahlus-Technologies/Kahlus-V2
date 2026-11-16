#!/usr/bin/env bash
# Build the package for the A100 operator from one pinned commit.
#
#   scripts/make_handoff.sh [COMMIT] [OUT_DIR]        # defaults: HEAD, dist/
#
# Writes OUT_DIR/kahlus-a100-handoff-<sha12>.zip with one folder of the same name:
#   COMMIT_HASH.txt             the full commit
#   README_HANDOFF.md           what to run and what to send back
#   kahlus-v2-<sha12>.tar.gz    a shallow git checkout of the commit, plus HANDOFF_SHA256SUMS of every tracked file
#   SHA256SUMS                  checksums of the three files above
# Only the commit goes in, never the working tree, so uncommitted files and secrets stay out.
set -euo pipefail

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)
sha=$(git -C "$repo" rev-parse --verify "${1:-HEAD}^{commit}")
short=${sha:0:12}
name=kahlus-a100-handoff-$short
out=${2:-$repo/dist}
mkdir -p "$out"
out=$(cd "$out" && pwd -P)

if command -v sha256sum >/dev/null; then sha256=(sha256sum); else sha256=(shasum -a 256); fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
src=$tmp/kahlus-v2
git -c init.defaultBranch=main init -q "$src"
git -C "$src" fetch -q --depth 1 "file://$repo" "$sha"
git -C "$src" -c advice.detachedHead=false checkout -q "$sha"
rm -f "$src/.git/FETCH_HEAD"  # it names this machine's path
(cd "$src" && git ls-files -z | xargs -0 "${sha256[@]}" > "$tmp/HANDOFF_SHA256SUMS")
mv "$tmp/HANDOFF_SHA256SUMS" "$src/"

pkg=$tmp/$name
mkdir "$pkg"
echo "$sha" > "$pkg/COMMIT_HASH.txt"
git -C "$repo" show "$sha:README_HANDOFF.md" > "$pkg/README_HANDOFF.md"
tar -C "$tmp" -czf "$pkg/kahlus-v2-$short.tar.gz" kahlus-v2
(cd "$pkg" && "${sha256[@]}" COMMIT_HASH.txt README_HANDOFF.md "kahlus-v2-$short.tar.gz" > SHA256SUMS)

rm -f "$out/$name.zip"
(cd "$tmp" && python3 -m zipfile -c "$out/$name.zip" "$name")
echo "$out/$name.zip"
"${sha256[@]}" "$out/$name.zip"
