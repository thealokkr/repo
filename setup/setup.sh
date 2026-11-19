#!/usr/bin/env bash
# Setup script for CloudServe Support Automation Capstone.
# Supports both executing (`bash setup/setup.sh`) and sourcing (`source setup/setup.sh`)
# in bash or zsh so the virtual environment remains active when sourced.

_cloudserve_is_sourced() {
  if [ -n "${ZSH_VERSION:-}" ]; then
    case "${ZSH_EVAL_CONTEXT:-}" in
      *:file*) return 0 ;;
    esac
  elif [ -n "${BASH_VERSION:-}" ]; then
    if [ "${BASH_SOURCE[0]:-}" != "$0" ]; then
      return 0
    fi
  fi
  return 1
}

_cloudserve_script_dir() {
  _src=""
  if [ -n "${ZSH_VERSION:-}" ]; then
    eval '_src="${(%):-%x}"'
  elif [ -n "${BASH_VERSION:-}" ]; then
    _src="${BASH_SOURCE[0]:-$0}"
  else
    _src="$0"
  fi
  (cd "$(dirname "$_src")" 2>/dev/null && pwd)
}

_cloudserve_is_repo_root() {
  _dir="$1"
  [ -f "$_dir/pyproject.toml" ] && [ -d "$_dir/src/cloudserve_support" ]
}

_cloudserve_update_env_key() {
  _env_file="$1"
  _key_name="$2"
  _key_val="$3"
  _tmp_file="${_env_file}.tmp.$$"
  _found=0
  : > "$_tmp_file"
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
      "${_key_name}="*)
        printf '%s=%s\n' "$_key_name" "$_key_val" >> "$_tmp_file"
        _found=1
        ;;
      *)
        printf '%s\n' "$_line" >> "$_tmp_file"
        ;;
    esac
  done < "$_env_file"
  if [ "$_found" -eq 0 ]; then
    printf '%s=%s\n' "$_key_name" "$_key_val" >> "$_tmp_file"
  fi
  mv "$_tmp_file" "$_env_file"
}

_cloudserve_get_env_value() {
  _env_file="$1"
  _key_name="$2"
  [ -f "$_env_file" ] || return 1
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in
      "${_key_name}="*)
        printf '%s' "${_line#*=}"
        return 0
        ;;
    esac
  done < "$_env_file"
  return 1
}

_cloudserve_usage() {
  cat <<'EOF'
Usage: setup.sh [OPTIONS]

Configure the local CloudServe Support Automation repository checkout: install uv
and Python 3.13 if missing, create .env from .env.example, prompt for
GEMINI_API_KEY (optional), sync Python/Chroma dependencies, activate .venv, and
print CLI commands.

Options:
      --reinit-db        Re-initialize SQLite DB, output schemas, and Chroma vector store
      --non-interactive  Skip the interactive GEMINI_API_KEY prompt
  -h, --help             Show this help message

Tip: Run `source setup/setup.sh` to keep the .venv virtual environment active
in your current shell after setup finishes.
EOF
}

_cloudserve_setup_main() {
  _reinit_db=0
  _non_interactive="${CLOUDSERVE_NONINTERACTIVE:-0}"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --reinit-db)
        _reinit_db=1
        shift
        ;;
      --non-interactive)
        _non_interactive=1
        shift
        ;;
      -h|--help)
        _cloudserve_usage
        return 0
        ;;
      *)
        echo "Error: Unknown option '$1'. Run with --help for usage." >&2
        return 1
        ;;
    esac
  done

  echo "==> CloudServe Support Automation environment setup"

  # 1. Locate repository root (must be run from an existing checkout)
  _script_dir="$(_cloudserve_script_dir)"
  _parent_dir="$(cd "$_script_dir/.." 2>/dev/null && pwd)"
  _repo_root=""

  if _cloudserve_is_repo_root "$(pwd)"; then
    _repo_root="$(pwd)"
    echo "==> Using current repository checkout: $_repo_root"
  elif _cloudserve_is_repo_root "$_parent_dir"; then
    _repo_root="$_parent_dir"
    echo "==> Using existing repository checkout: $_repo_root"
  else
    echo "Error: Could not locate the CloudServe repository root (pyproject.toml and src/cloudserve_support)." >&2
    echo "Run setup/setup.sh from inside your cloned repository checkout." >&2
    return 1
  fi

  cd "$_repo_root" || return 1

  # 2. Install uv if missing
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "==> Installing uv..."
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | sh || return 1
    else
      echo "Error: curl or wget is required to install uv." >&2
      return 1
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
      echo "Error: uv was installed but is not on PATH. Add \$HOME/.local/bin to PATH and retry." >&2
      return 1
    fi
  else
    echo "==> Found uv: $(uv --version)"
  fi

  # 3. Check required Python version (from .python-version) and install only if missing
  _required_python="3.13"
  if [ -f ".python-version" ]; then
    _required_python="$(tr -d '[:space:]' < .python-version)"
  fi
  _found_python="$(uv python find "$_required_python" 2>/dev/null || true)"
  if [ -n "$_found_python" ]; then
    echo "==> Found Python ($_required_python): $_found_python"
  else
    echo "==> Python $_required_python not found; installing with uv..."
    uv python install "$_required_python" || return 1
  fi

  # 4. Copy .env.example to .env and prompt for GEMINI_API_KEY (blank allowed)
  if [ ! -f ".env.example" ]; then
    echo "Error: .env.example not found in $_repo_root" >&2
    return 1
  fi

  if [ ! -f ".env" ]; then
    cp .env.example .env || return 1
    echo "==> Created .env from .env.example"
  else
    echo "==> Found existing .env (keeping existing settings)"
  fi

  _existing_gemini_key="$(_cloudserve_get_env_value ".env" "GEMINI_API_KEY" || true)"
  _existing_google_key="$(_cloudserve_get_env_value ".env" "GOOGLE_API_KEY" || true)"

  if [ "$_non_interactive" != "1" ] && { [ -t 0 ] || [ -r /dev/tty ]; }; then
    echo ""
    if [ -n "$_existing_gemini_key" ] || [ -n "$_existing_google_key" ]; then
      printf "GEMINI_API_KEY/GOOGLE_API_KEY is already configured in .env.\n"
      printf "Enter a new GEMINI_API_KEY to overwrite, or press Enter to keep existing: "
    else
      printf "Enter your GEMINI_API_KEY for .env (or press Enter to leave blank): "
    fi
    _entered_key=""
    if [ -t 0 ]; then
      if IFS= read -r -s _entered_key; then
        printf "\n"
      else
        printf "\n"
        _entered_key=""
      fi
    else
      if IFS= read -r -s _entered_key < /dev/tty; then
        printf "\n"
      else
        printf "\n"
        _entered_key=""
      fi
    fi
    if [ -n "$_entered_key" ]; then
      _cloudserve_update_env_key ".env" "GEMINI_API_KEY" "$_entered_key"
      echo "==> Saved GEMINI_API_KEY to .env"
    elif [ -z "$_existing_gemini_key" ] && [ -z "$_existing_google_key" ]; then
      echo "==> Left GEMINI_API_KEY blank in .env (setup completed; add GEMINI_API_KEY or GOOGLE_API_KEY to .env before running cloudserve run)."
    else
      echo "==> Kept existing API key in .env."
    fi
    unset _entered_key
  else
    echo "==> Non-interactive environment detected; skipping GEMINI_API_KEY prompt."
  fi

  # 5. Sync Python packages (Chroma, google-genai, jsonschema, pytest, cloudserve CLI)
  echo "==> Syncing Python packages and creating .venv with uv sync --frozen..."
  uv sync --frozen || return 1

  # 6. Activate the virtual environment
  if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . ".venv/bin/activate" || return 1
    echo "==> Activated virtual environment: $(command -v python)"
  elif [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    . ".venv/Scripts/activate" || return 1
    echo "==> Activated virtual environment: $(command -v python)"
  else
    echo "Error: Virtual environment activation script not found under .venv/" >&2
    return 1
  fi

  # 7. Verify or re-initialize checked-in vector store and SQLite DB
  if [ "$_reinit_db" -eq 1 ]; then
    echo "==> Re-initializing SQLite DB, output schemas, and Chroma vector store..."
    python scripts/init_runtime_db.py || return 1
    python scripts/init_output_schemas.py || return 1
    python scripts/init_vector_store.py || return 1
    _final_gemini_key="$(_cloudserve_get_env_value ".env" "GEMINI_API_KEY" || true)"
    _final_google_key="$(_cloudserve_get_env_value ".env" "GOOGLE_API_KEY" || true)"
    if [ -n "$_final_gemini_key" ] || [ -n "$_final_google_key" ]; then
      echo "==> API key present; running KB ingestion into Chroma..."
      uv run --env-file .env python scripts/ingest_kb.py || return 1
    else
      echo "==> No API key set; skipped scripts/ingest_kb.py (run it after adding GEMINI_API_KEY to .env)."
    fi
  else
    echo "==> Checking pre-packaged vector store and SQLite database..."
    if [ -f "data/index/chroma/chroma.sqlite3" ] && [ -f "data/index/manifest.json" ]; then
      python -c '
import chromadb
client = chromadb.PersistentClient(path="data/index/chroma")
col = client.get_collection("cloudserve_kb")
print(f"    Chroma vector store ready: collection={col.name}, passages={col.count()}")
' || return 1
    else
      echo "    Warning: Checked-in Chroma vector store not found at data/index/chroma/."
      echo "    Run: uv run python scripts/init_vector_store.py && uv run --env-file .env python scripts/ingest_kb.py"
    fi

    if [ -f "data/runtime/cloudserve.sqlite3" ]; then
      python -c '
import sqlite3
conn = sqlite3.connect("data/runtime/cloudserve.sqlite3")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type=\"table\"").fetchall()]
print(f"    SQLite runtime DB ready: data/runtime/cloudserve.sqlite3 (tables={tables})")
conn.close()
' || return 1
    else
      echo "    Warning: Checked-in SQLite database not found at data/runtime/cloudserve.sqlite3."
      echo "    Run: uv run python scripts/init_runtime_db.py"
    fi
  fi

  PYTHONWARNINGS=ignore::SyntaxWarning cloudserve --help >/dev/null || return 1

  echo ""
  echo "========================================================================"
  echo "Setup complete!"
  echo "Repository: $_repo_root"
  echo "Environment: $_repo_root/.venv"
  echo "Config file: $_repo_root/.env"
  echo ""
  if ! _cloudserve_is_sourced; then
    echo "Activate the virtual environment in your shell:"
    echo "  cd \"$_repo_root\" && source .venv/bin/activate"
    echo ""
  fi
  echo "Run the CLI (see available commands):"
  echo "  uv run cloudserve --help"
  echo ""
  echo "Example: process the validation ticket set:"
  echo "  uv run cloudserve run --input data/tickets/validation_tickets.json --skip-draft --skip-draft-review"
  echo ""
  echo "(Or with .venv activated, simply run:)"
  echo "  cloudserve run --input data/tickets/validation_tickets.json --skip-draft --skip-draft-review"
  echo "========================================================================"
  return 0
}

_cloudserve_setup_main "$@"
_cloudserve_rc=$?
if _cloudserve_is_sourced; then
  return "$_cloudserve_rc"
else
  exit "$_cloudserve_rc"
fi
