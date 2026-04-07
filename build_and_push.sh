#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="${DIST_DIR:-dist}"
S3_BUCKET="${S3_BUCKET:-curve-fit.hexq.ru}"
S3_PREFIX="${S3_PREFIX:-}"
S3_PREFIX="${S3_PREFIX#/}"
S3_PREFIX="${S3_PREFIX%/}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' is not available in PATH." >&2
    exit 1
  fi
}

content_type_for() {
  local file_path="$1"
  case "$file_path" in
    *.html) echo "text/html; charset=utf-8" ;;
    *.js) echo "text/javascript; charset=utf-8" ;;
    *.css) echo "text/css; charset=utf-8" ;;
    *.wasm) echo "application/wasm" ;;
    *.svg) echo "image/svg+xml" ;;
    *.png) echo "image/png" ;;
    *.jpg | *.jpeg) echo "image/jpeg" ;;
    *.webp) echo "image/webp" ;;
    *.json) echo "application/json" ;;
    *.txt) echo "text/plain; charset=utf-8" ;;
    *.ico) echo "image/x-icon" ;;
    *) echo "application/octet-stream" ;;
  esac
}

target_base_uri() {
  if [[ -n "$S3_PREFIX" ]]; then
    echo "s3://${S3_BUCKET}/${S3_PREFIX}"
  else
    echo "s3://${S3_BUCKET}"
  fi
}

upload_file() {
  local src_path="$1"
  local rel_path="$2"
  local object_key="$rel_path"
  local content_type
  local target_uri

  if [[ -n "$S3_PREFIX" ]]; then
    object_key="${S3_PREFIX}/${rel_path}"
  fi

  content_type="$(content_type_for "$rel_path")"
  target_uri="s3://${S3_BUCKET}/${object_key}"

  if [[ "$rel_path" == *.wasm ]]; then
    local gzip_tmp="${src_path}.gz"
    gzip -9 -c "$src_path" >"$gzip_tmp"
    yc storage s3 cp "$gzip_tmp" "$target_uri" \
      --content-type "$content_type" \
      --content-encoding gzip
    rm -f "$gzip_tmp"
  else
    yc storage s3 cp "$src_path" "$target_uri" --content-type "$content_type"
  fi

  echo "Uploaded: ${rel_path} -> ${target_uri}"
}

require_command trunk
require_command yc
require_command gzip
require_command find

echo "Building web release with trunk..."
trunk build --release

if [[ ! -d "$DIST_DIR" ]]; then
  echo "Error: dist directory '$DIST_DIR' does not exist after build." >&2
  exit 1
fi

mapfile -d '' dist_files < <(find "$DIST_DIR" -type f -print0 | sort -z)
if [[ "${#dist_files[@]}" -eq 0 ]]; then
  echo "Error: no files found in '$DIST_DIR'." >&2
  exit 1
fi

regular_files=()
html_files=()

for src_path in "${dist_files[@]}"; do
  rel_path="${src_path#"$DIST_DIR"/}"
  if [[ "$rel_path" == *.html ]]; then
    html_files+=("$src_path")
  else
    regular_files+=("$src_path")
  fi
done

TARGET_BASE_URI="$(target_base_uri)"
echo "Cleaning remote objects at ${TARGET_BASE_URI}..."
yc storage s3 rm "${TARGET_BASE_URI}" --recursive

echo "Uploading static assets to ${TARGET_BASE_URI}..."
for src_path in "${regular_files[@]}"; do
  rel_path="${src_path#"$DIST_DIR"/}"
  upload_file "$src_path" "$rel_path"
done

for src_path in "${html_files[@]}"; do
  rel_path="${src_path#"$DIST_DIR"/}"
  upload_file "$src_path" "$rel_path"
done

echo "Publish complete."
