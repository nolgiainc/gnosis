#!/usr/bin/env bash
#
# Container image vulnerability gate (sec-platform lane, SOC 2 CC7.1).
#
#   .github/scripts/image-scan.sh <image-ref> [<label>]
#
# Runs Trivy from its official image, pinned by digest, against <image-ref>:
#
#   1. gate    CRITICAL vulnerabilities that HAVE a fixed version -> exit 1
#   2. report  HIGH vulnerabilities (fixed or not) -> job summary, never fails
#
# Vulnerability scanner only (`--scanners vuln`); secrets are gitleaks' job.
#
# Accepted risks live in .trivyignore.yaml at the repository root. Every entry
# carries an id, a statement (why it is not fixable or not reachable here) and
# an `expired_at` no more than 90 days out, so an exception lapses back into a
# red build instead of living forever. A CRITICAL with a fix is fixed (bump the
# base image digest or the package), never ignored.
#
# The image is read through the local Docker daemon. A ref that is not in the
# local store is pulled first with the runner's own registry credentials
# (docker login or the gcloud credential helper), so the Trivy container itself
# holds no credentials.

set -euo pipefail

ref="${1:?usage: image-scan.sh <image-ref> [label]}"
label="${2:-$ref}"

# Bump deliberately. Resolve the digest of a new release with
#   docker buildx imagetools inspect aquasec/trivy:<version>
TRIVY_IMAGE="${TRIVY_IMAGE:-aquasec/trivy:0.75.0@sha256:af6acf9a6b85dfe389a1941505c0ce9efef52a4719635e1a962f022a3d855daa}"

root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cache="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/trivy-cache"
summary="${GITHUB_STEP_SUMMARY:-/dev/null}"
mkdir -p "$cache"

# Run as the invoking user (not root) so the DB cache stays removable by the
# runner; the docker socket's group grants daemon access.
run=(docker run --rm
  --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /var/run/docker.sock)"
  -e TRIVY_CACHE_DIR=/cache -v "$cache:/cache"
  -v /var/run/docker.sock:/var/run/docker.sock)
opts=(image --scanners vuln --image-src docker --cache-backend memory
  --table-mode detailed --no-progress)
if [ -f "$root/.trivyignore.yaml" ]; then
  # Policy check: every entry needs a statement and an expiry at most 90 days
  # out. Trivy itself would accept an entry with neither.
  if ! awk -v max="$(date -u -d '+90 days' +%Y-%m-%d)" '
      function finish() {
        if (id == "") return
        if (!st) { print "  " id ": missing statement"; bad = 1 }
        if (until == "") { print "  " id ": missing expired_at"; bad = 1 }
        else if (until > max) { print "  " id ": expired_at " until " is more than 90 days out (latest allowed " max ")"; bad = 1 }
      }
      /^[[:space:]]*-[[:space:]]+id:/ { finish(); id = $NF; st = 0; until = ""; next }
      /^[[:space:]]+statement:/ { st = 1 }
      /^[[:space:]]+expired_at:/ { until = $NF; gsub(/["\047]/, "", until) }
      END { finish(); exit bad }' "$root/.trivyignore.yaml"; then
    echo "::error title=Image scan::.trivyignore.yaml violates the allowlist policy (see above)"
    exit 1
  fi
  run+=(-v "$root/.trivyignore.yaml:/trivyignore.yaml:ro")
  opts+=(--ignorefile /trivyignore.yaml)
fi

if ! docker image inspect "$ref" >/dev/null 2>&1; then
  echo "==> pulling $ref"
  docker pull --quiet "$ref" >/dev/null
fi

gate_out="$(mktemp)"
high_out="$(mktemp)"
trap 'rm -f "$gate_out" "$high_out"' EXIT

# Exit code 8 means "CRITICAL with a fix found"; any other non-zero status is
# a scanner failure (DB download, unreadable image), which also fails closed.
echo "==> gate: CRITICAL with a fix available ($label)"
gate=0
"${run[@]}" "$TRIVY_IMAGE" "${opts[@]}" \
  --severity CRITICAL --ignore-unfixed --exit-code 8 --format table \
  "$ref" >"$gate_out" || gate=$?
cat "$gate_out"
if [ "$gate" -ne 0 ] && [ "$gate" -ne 8 ]; then
  echo "::error title=Image scan::Trivy failed (exit $gate) scanning $label"
  exit 1
fi

echo "==> report: HIGH ($label)"
"${run[@]}" "$TRIVY_IMAGE" "${opts[@]}" --skip-db-update \
  --severity HIGH --exit-code 0 --format table \
  "$ref" >"$high_out"
cat "$high_out"

{
  echo "### Image scan: \`$label\`"
  echo
  if [ "$gate" -eq 8 ]; then
    echo "**FAILED**: CRITICAL vulnerabilities with a fixed version available. Fix them (bump the base image digest or the package); add a \`.trivyignore.yaml\` entry only when the fix is genuinely out of our hands."
  else
    echo "Passed: no CRITICAL vulnerability with a fix available."
  fi
  echo
  echo "<details><summary>CRITICAL (fixable)</summary>"
  echo
  echo '```text'
  if [ -s "$gate_out" ]; then head -c 300000 "$gate_out"; else echo "(none)"; fi
  echo '```'
  echo "</details>"
  echo
  echo "<details><summary>HIGH (report only)</summary>"
  echo
  echo '```text'
  if [ -s "$high_out" ]; then head -c 600000 "$high_out"; else echo "(none)"; fi
  echo '```'
  echo "</details>"
  echo
} >>"$summary"

if [ "$gate" -eq 8 ]; then
  echo "::error title=Image scan::$label has CRITICAL vulnerabilities with a fix available (see the job summary)"
  exit 1
fi
