#!/usr/bin/env bash
# Optional env for tone_take.py: the Montreal Forced Aligner (Kaldi, from conda-forge) plus Praat
# (praat-parselmouth) in their own conda env, and the Mandarin MFA models. About 1.5 GB; nothing global.
#
#   bash setup_tone.sh            # or: bash setup_env.sh --with-tone
#
# Lands in $SEV_WS (default ~/.cache/sglang-explainer-video): micromamba/, mfa-env/, mfa-models/.
# MFA's own `mfa model download` asks the GitHub API and hits its anonymous rate limit after a few calls,
# so the models come straight from the release assets.
set -euo pipefail

WS="${SEV_WS:-${SGLANG_VIDEO_HOME:-$HOME/.cache/sglang-explainer-video}}"
MM_DIR="$WS/micromamba"
ENV_DIR="${SEV_MFA_ENV:-$WS/mfa-env}"
MODELS="${SEV_MFA_MODELS:-$WS/mfa-models}"
MFA_VERSION="${MFA_VERSION:-3.4.3}"
REL="https://github.com/MontrealCorpusTools/mfa-models/releases/download"
mkdir -p "$MM_DIR" "$MODELS"

# 1. micromamba (a single binary) for the conda-forge packages MFA needs (Kaldi, OpenFst).
MM="$MM_DIR/bin/micromamba"
if [ ! -x "$MM" ]; then
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) plat=osx-arm64 ;;
    Darwin/x86_64) plat=osx-64 ;;
    Linux/aarch64) plat=linux-aarch64 ;;
    Linux/x86_64) plat=linux-64 ;;
    *) echo "✗ no micromamba build for $(uname -s)/$(uname -m)" >&2; exit 1 ;;
  esac
  curl -fsSL "https://micro.mamba.pm/api/micromamba/$plat/latest" | tar -xj -C "$MM_DIR" bin/micromamba
fi

# 2. MFA + Praat in one env. pip cannot install MFA usefully: its Kaldi bindings are conda-forge only.
if [ ! -x "$ENV_DIR/bin/mfa" ]; then
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$MM_DIR/root}" "$MM" create -y -q -p "$ENV_DIR" -c conda-forge \
    "montreal-forced-aligner=$MFA_VERSION" python=3.11
fi
"$ENV_DIR/bin/python" -c "import parselmouth" 2>/dev/null || "$ENV_DIR/bin/python" -m pip install -q praat-parselmouth

# 3. Mandarin acoustic model (v3.0.0, tone-marked IPA phones) and its pronunciation dictionary.
[ -s "$MODELS/mandarin_mfa.zip" ] || curl -fsSL -o "$MODELS/mandarin_mfa.zip" "$REL/acoustic-mandarin_mfa-v3.0.0/mandarin_mfa.zip"
[ -s "$MODELS/mandarin_china_mfa.dict" ] ||
  curl -fsSL -o "$MODELS/mandarin_china_mfa.dict" "$REL/dictionary-mandarin_china_mfa-v3.0.0/mandarin_china_mfa.dict"

PATH="$ENV_DIR/bin:$PATH" MFA_ROOT_DIR="$WS/mfa-root" "$ENV_DIR/bin/mfa" version >/dev/null
echo "✓ tone env ready: $ENV_DIR (MFA $("$ENV_DIR/bin/mfa" version 2>/dev/null | tail -1)), models in $MODELS"
