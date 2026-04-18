#!/usr/bin/env bash
set -euo pipefail

# Extract only the fields needed for stage-1 conversion from .fchk and pair LIDI .txt files.
#
# Output layout:
#   <out-dir>/
#     fchk/<stem>.fchk   (minimal fchk: Number of atoms, Charge, Multiplicity,
#                         Atomic numbers, Current cartesian coordinates, Mulliken Charges)
#     lidi/<stem>.txt    (paired LIDI txt; copied as-is by default)
#     manifest.tsv
#
# Example:
#   bash scripts/extract_minimal_fchk_lidi.sh \
#     --fchk-dir /data/RGD1_fchk \
#     --lidi-dir /data/RGD1_LIDI \
#     --out-dir /data/RGD1_minimal \
#     --archive
#
# Then on another machine:
#   python scripts/convert_fchk_lidi_to_pretrain_h5.py \
#     --fchk-dir /data/RGD1_minimal/fchk \
#     --lidi-dir /data/RGD1_minimal/lidi \
#     --out-prefix /data/rgd1_stage1

usage() {
  cat <<'EOF'
Usage:
  extract_minimal_fchk_lidi.sh --fchk-dir DIR --lidi-dir DIR --out-dir DIR [options]

Required:
  --fchk-dir DIR         Source directory for .fchk files
  --lidi-dir DIR         Source directory for LIDI .txt files
  --out-dir DIR          Output root directory

Options:
  --fchk-pattern PAT     Default: *.fchk
  --lidi-pattern PAT     Default: *.txt
  --no-recursive         Disable recursive search
  --trim-lidi            Keep only section from "Total delocalization index matrix" onward
  --max-errors N         Max skipped files before abort (default: 0 means abort on first error)
  --archive              Create <out-dir>.tar.gz after extraction
  --gzip-level N         gzip level for archive (default: 6)
  --overwrite            Remove existing out-dir before extraction
  -h, --help             Show this help
EOF
}

FCHK_DIR=""
LIDI_DIR=""
OUT_DIR=""
FCHK_PATTERN="*.fchk"
LIDI_PATTERN="*.txt"
RECURSIVE=1
TRIM_LIDI=0
MAX_ERRORS=0
MAKE_ARCHIVE=0
GZIP_LEVEL=6
OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fchk-dir) FCHK_DIR="${2:-}"; shift 2 ;;
    --lidi-dir) LIDI_DIR="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --fchk-pattern) FCHK_PATTERN="${2:-}"; shift 2 ;;
    --lidi-pattern) LIDI_PATTERN="${2:-}"; shift 2 ;;
    --no-recursive) RECURSIVE=0; shift ;;
    --trim-lidi) TRIM_LIDI=1; shift ;;
    --max-errors) MAX_ERRORS="${2:-}"; shift 2 ;;
    --archive) MAKE_ARCHIVE=1; shift ;;
    --gzip-level) GZIP_LEVEL="${2:-}"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[error] unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$FCHK_DIR" || -z "$LIDI_DIR" || -z "$OUT_DIR" ]]; then
  echo "[error] --fchk-dir, --lidi-dir, --out-dir are required." >&2
  usage
  exit 2
fi

if [[ ! -d "$FCHK_DIR" ]]; then
  echo "[error] fchk dir not found: $FCHK_DIR" >&2
  exit 2
fi
if [[ ! -d "$LIDI_DIR" ]]; then
  echo "[error] LIDI dir not found: $LIDI_DIR" >&2
  exit 2
fi
if ! [[ "$MAX_ERRORS" =~ ^[0-9]+$ ]]; then
  echo "[error] --max-errors must be a non-negative integer." >&2
  exit 2
fi
if ! [[ "$GZIP_LEVEL" =~ ^[0-9]+$ ]] || (( GZIP_LEVEL < 1 || GZIP_LEVEL > 9 )); then
  echo "[error] --gzip-level must be integer in [1,9]." >&2
  exit 2
fi

OUT_DIR="$(cd "$(dirname "$OUT_DIR")" && pwd)/$(basename "$OUT_DIR")"
if [[ -e "$OUT_DIR" ]]; then
  if [[ "$OVERWRITE" -eq 1 ]]; then
    rm -rf "$OUT_DIR"
  else
    echo "[error] out-dir exists: $OUT_DIR (use --overwrite)" >&2
    exit 2
  fi
fi

mkdir -p "$OUT_DIR/fchk" "$OUT_DIR/lidi"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

build_index() {
  local src_dir="$1"
  local pattern="$2"
  local out_tsv="$3"
  local recursive="$4"

  if [[ "$recursive" -eq 1 ]]; then
    find "$src_dir" -type f -name "$pattern" -print \
      | awk -F/ '
          {
            name = $NF
            sub(/\.[^.]+$/, "", name)
            print name "\t" $0
          }
        ' \
      | LC_ALL=C sort -t $'\t' -k1,1 > "$out_tsv"
  else
    find "$src_dir" -maxdepth 1 -type f -name "$pattern" -print \
      | awk -F/ '
          {
            name = $NF
            sub(/\.[^.]+$/, "", name)
            print name "\t" $0
          }
        ' \
      | LC_ALL=C sort -t $'\t' -k1,1 > "$out_tsv"
  fi
}

check_duplicates() {
  local tsv="$1"
  local tag="$2"
  local dup
  dup="$(cut -f1 "$tsv" | uniq -d | head -n 1 || true)"
  if [[ -n "$dup" ]]; then
    echo "[error] duplicate $tag stem detected: $dup" >&2
    exit 2
  fi
}

extract_min_fchk() {
  local src="$1"
  local dst="$2"
  local tmp="${dst}.tmp"

  awk '
    function trim(s) {
      sub(/^[[:space:]]+/, "", s)
      sub(/[[:space:]]+$/, "", s)
      return s
    }
    function parse_n_from_meta(s,   t) {
      t = s
      sub(/^.*N=[[:space:]]*/, "", t)
      if (t == s) return -1
      sub(/[^0-9].*$/, "", t)
      if (t == "") return -1
      return t + 0
    }

    BEGIN {
      in_block = 0
      remain = 0
      have_n_atoms = 0
      have_charge = 0
      have_mult = 0
      have_z = 0
      have_coords = 0
      have_mulliken = 0
      FS = " "
    }

    {
      if (in_block) {
        print $0
        remain -= NF
        if (remain <= 0) {
          in_block = 0
          remain = 0
        }
        next
      }

      name = trim(substr($0, 1, 43))
      meta = substr($0, 44)

      if (name == "Number of atoms") {
        print $0
        have_n_atoms = 1
        next
      }
      if (name == "Charge") {
        print $0
        have_charge = 1
        next
      }
      if (name == "Multiplicity") {
        print $0
        have_mult = 1
        next
      }

      if (name == "Atomic numbers" || name == "Current cartesian coordinates" || name == "Mulliken Charges") {
        print $0
        remain = parse_n_from_meta(meta)
        if (remain >= 0) {
          in_block = 1
          if (name == "Atomic numbers") have_z = 1
          else if (name == "Current cartesian coordinates") have_coords = 1
          else if (name == "Mulliken Charges") have_mulliken = 1
        } else {
          printf("[awk-error] cannot parse N= for \"%s\"\n", name) > "/dev/stderr"
          exit 3
        }
        next
      }
    }

    END {
      if (in_block) {
        print "[awk-error] unfinished array block at EOF" > "/dev/stderr"
        exit 4
      }
      if (!have_n_atoms || !have_charge || !have_mult || !have_z || !have_coords) {
        print "[awk-error] missing required fields in fchk" > "/dev/stderr"
        exit 5
      }
      # Mulliken Charges is optional; converter can fallback to zeros if missing.
    }
  ' "$src" > "$tmp"

  mv "$tmp" "$dst"
  [[ -s "$dst" ]]
}

extract_or_copy_lidi() {
  local src="$1"
  local dst="$2"
  local trim="$3"
  local tmp="${dst}.tmp"

  if [[ "$trim" -eq 0 ]]; then
    cp "$src" "$tmp"
    mv "$tmp" "$dst"
    [[ -s "$dst" ]]
    return
  fi

  awk '
    BEGIN { keep = 0; found = 0 }
    {
      if (!keep && index($0, "Total delocalization index matrix") > 0) {
        keep = 1
        found = 1
      }
      if (keep) print $0
    }
    END {
      if (!found) {
        print "[awk-error] LIDI header not found" > "/dev/stderr"
        exit 6
      }
    }
  ' "$src" > "$tmp"
  mv "$tmp" "$dst"
  [[ -s "$dst" ]]
}

echo "[scan] building indices ..."
build_index "$FCHK_DIR" "$FCHK_PATTERN" "$TMP_DIR/fchk.tsv" "$RECURSIVE"
build_index "$LIDI_DIR" "$LIDI_PATTERN" "$TMP_DIR/lidi.tsv" "$RECURSIVE"

if [[ ! -s "$TMP_DIR/fchk.tsv" ]]; then
  echo "[error] no fchk files found." >&2
  exit 2
fi
if [[ ! -s "$TMP_DIR/lidi.tsv" ]]; then
  echo "[error] no LIDI txt files found." >&2
  exit 2
fi

check_duplicates "$TMP_DIR/fchk.tsv" "fchk"
check_duplicates "$TMP_DIR/lidi.tsv" "lidi"

join -t $'\t' "$TMP_DIR/fchk.tsv" "$TMP_DIR/lidi.tsv" > "$TMP_DIR/pairs.tsv" || true
if [[ ! -s "$TMP_DIR/pairs.tsv" ]]; then
  echo "[error] no paired files found by stem." >&2
  exit 2
fi

fchk_total="$(wc -l < "$TMP_DIR/fchk.tsv" | tr -d ' ')"
lidi_total="$(wc -l < "$TMP_DIR/lidi.tsv" | tr -d ' ')"
pair_total="$(wc -l < "$TMP_DIR/pairs.tsv" | tr -d ' ')"
echo "[scan] fchk=$fchk_total  lidi=$lidi_total  paired=$pair_total"

manifest="$OUT_DIR/manifest.tsv"
printf "stem\tminimal_fchk\tminimal_lidi\tsource_fchk\tsource_lidi\n" > "$manifest"

ok=0
err=0
while IFS=$'\t' read -r stem fchk_src lidi_src; do
  out_fchk="$OUT_DIR/fchk/${stem}.fchk"
  out_lidi="$OUT_DIR/lidi/${stem}.txt"
  mkdir -p "$(dirname "$out_fchk")" "$(dirname "$out_lidi")"

  if extract_min_fchk "$fchk_src" "$out_fchk" && extract_or_copy_lidi "$lidi_src" "$out_lidi" "$TRIM_LIDI"; then
    printf "%s\t%s\t%s\t%s\t%s\n" "$stem" "$out_fchk" "$out_lidi" "$fchk_src" "$lidi_src" >> "$manifest"
    ok=$((ok + 1))
  else
    err=$((err + 1))
    echo "[warn] failed: $stem" >&2
    rm -f "$out_fchk" "$out_lidi"
    if [[ "$MAX_ERRORS" -eq 0 || "$err" -gt "$MAX_ERRORS" ]]; then
      echo "[error] abort after $err failures." >&2
      exit 1
    fi
  fi
done < "$TMP_DIR/pairs.tsv"

echo "[done] extracted=$ok  failed=$err"
du -sh "$OUT_DIR/fchk" "$OUT_DIR/lidi" "$OUT_DIR" 2>/dev/null || true

if [[ "$MAKE_ARCHIVE" -eq 1 ]]; then
  archive="${OUT_DIR}.tar.gz"
  echo "[archive] writing $archive ..."
  tar -C "$(dirname "$OUT_DIR")" -cf - "$(basename "$OUT_DIR")" \
    | gzip -"${GZIP_LEVEL}" > "$archive"
  du -sh "$archive" 2>/dev/null || true
fi
