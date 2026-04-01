#!/usr/bin/env bash
set -euo pipefail

base_url="${1:-https://cudacourseh100.github.io}"
base_url="${base_url%/}"
max_attempts="${MAX_ATTEMPTS:-12}"
sleep_seconds="${SLEEP_SECONDS:-10}"

last_error=""

trim_cr() {
  tr -d '\r'
}

header_value() {
  local headers="$1"
  local name="$2"

  printf '%s\n' "$headers" | awk -v key="$name" '
    BEGIN { IGNORECASE = 1 }
    $0 ~ ("^" key ":") { value = $0 }
    END {
      sub(/^[^:]+:[[:space:]]*/, "", value)
      print value
    }
  ' | trim_cr
}

status_code() {
  local headers="$1"

  printf '%s\n' "$headers" | awk '
    toupper($0) ~ /^HTTP\// { code = $2 }
    END { print code }
  ' | trim_cr
}

fetch_headers() {
  local url="$1"
  curl -sSIL "$url"
}

fetch_body() {
  local url="$1"
  curl -fsSL "$url"
}

assert_status_and_type() {
  local url="$1"
  local expected_status="$2"
  local expected_type="$3"
  local headers status content_type

  headers="$(fetch_headers "$url")" || {
    last_error="Failed to fetch headers for $url"
    return 1
  }

  status="$(status_code "$headers")"
  content_type="$(header_value "$headers" "content-type")"

  if [[ "$status" != "$expected_status" ]]; then
    last_error="Expected HTTP $expected_status for $url but got $status"
    return 1
  fi

  if [[ "$content_type" != *"$expected_type"* ]]; then
    last_error="Expected content type containing '$expected_type' for $url but got '$content_type'"
    return 1
  fi
}

assert_body_contains() {
  local url="$1"
  local expected_fragment="$2"
  local body

  body="$(fetch_body "$url")" || {
    last_error="Failed to fetch body for $url"
    return 1
  }

  if ! grep -Fq "$expected_fragment" <<<"$body"; then
    last_error="Expected body from $url to contain: $expected_fragment"
    return 1
  fi
}

run_checks() {
  assert_status_and_type "$base_url/" "200" "text/html"
  assert_body_contains "$base_url/" "<link rel=\"canonical\" href=\"$base_url/\">"

  assert_status_and_type "$base_url/robots.txt" "200" "text/plain"
  assert_body_contains "$base_url/robots.txt" "Sitemap: $base_url/sitemap.xml"

  assert_status_and_type "$base_url/sitemap.xml" "200" "application/xml"
  assert_body_contains "$base_url/sitemap.xml" "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"
  assert_body_contains "$base_url/sitemap.xml" "<loc>$base_url/</loc>"

  assert_status_and_type "$base_url/sitemap" "200" "application/xml"
  assert_body_contains "$base_url/sitemap" "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"
}

attempt=1
while (( attempt <= max_attempts )); do
  echo "Verifying live SEO endpoints ($attempt/$max_attempts)..."

  if run_checks; then
    echo "Live SEO endpoints look healthy."
    exit 0
  fi

  echo "$last_error" >&2

  if (( attempt < max_attempts )); then
    echo "Waiting ${sleep_seconds}s for GitHub Pages propagation..."
    sleep "$sleep_seconds"
  fi

  (( attempt++ ))
done

echo "Live SEO verification failed after $max_attempts attempts." >&2
exit 1
