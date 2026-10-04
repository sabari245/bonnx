#!/usr/bin/env bash
# onnxviz installer — detects your system, downloads the latest release and installs it.
#   curl -fsSL https://raw.githubusercontent.com/sabari245/onnxviz/main/setup.sh | bash
# Options (pass after `bash -s --` when piping):
#   --version vX.Y.Z   install a specific release instead of the latest
#   --appimage         force the user-local AppImage install (no root needed)
#   --uninstall        remove what this script installed
set -euo pipefail

REPO="sabari245/onnxviz"
VERSION="latest"
FORCE_APPIMAGE=0
UNINSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="${2:?--version needs a value}"; shift 2 ;;
    --appimage) FORCE_APPIMAGE=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
if [ -t 2 ]; then PROG="--progress-bar"; else PROG="-sS"; fi

BIN_DIR="${HOME}/.local/bin"
APP_DIR="${HOME}/.local/share/onnxviz"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR="${HOME}/.local/share/icons/hicolor/512x512/apps"

if [ "$UNINSTALL" = 1 ]; then
  if have dpkg && dpkg -s onnxviz >/dev/null 2>&1; then
    say "Removing the onnxviz package"
    if [ "$(id -u)" -eq 0 ]; then apt-get remove -y onnxviz; else sudo apt-get remove -y onnxviz; fi
  fi
  rm -rf "$APP_DIR" "$BIN_DIR/onnxviz" "$DESKTOP_DIR/onnxviz.desktop" "$ICON_DIR/onnxviz.png"
  have update-desktop-database && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
  say "Uninstalled."
  exit 0
fi

# ── system detection ──────────────────────────────────────────────────────────
[ "$(uname -s)" = "Linux" ] || die "Only Linux is supported (detected $(uname -s))."
case "$(uname -m)" in
  x86_64|amd64) ARCH="x86_64"; DEB_ARCH="amd64" ;;
  *) die "Only x86_64 builds are published (detected $(uname -m))." ;;
esac
have curl || die "curl is required."

DISTRO="unknown"
[ -r /etc/os-release ] && DISTRO="$(. /etc/os-release && echo "${PRETTY_NAME:-${NAME:-unknown}}")"
say "Detected: $DISTRO ($(uname -m))"

# ── find the release ──────────────────────────────────────────────────────────
if [ "$VERSION" = "latest" ]; then API="https://api.github.com/repos/$REPO/releases/latest"
else API="https://api.github.com/repos/$REPO/releases/tags/$VERSION"; fi
JSON="$(curl -fsSL -H 'Accept: application/vnd.github+json' "$API")" || die "Could not fetch release info from $API"
TAG="$(printf '%s' "$JSON" | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')"
[ -n "$TAG" ] || die "Could not determine the release tag."
say "Release: $TAG"

asset_url() { printf '%s' "$JSON" | grep '"browser_download_url"' | sed -E 's/.*"(https[^"]+)".*/\1/' | grep -E -e "$1" | head -n1; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

fetch() { # url -> file in $TMP, verified against SHA256SUMS.txt when present
  local url="$1" file="$TMP/$(basename "$1")"
  say "Downloading $(basename "$url")"
  curl -fL $PROG -o "$file" "$url" || die "Download failed: $url"
  local sums; sums="$(asset_url 'SHA256SUMS\.txt$' || true)"
  if [ -n "$sums" ] && have sha256sum; then
    curl -fsSL -o "$TMP/SHA256SUMS.txt" "$sums"
    local want; want="$(grep " $(basename "$file")\$" "$TMP/SHA256SUMS.txt" | cut -d' ' -f1)"
    if [ -n "$want" ]; then
      [ "$(sha256sum "$file" | cut -d' ' -f1)" = "$want" ] || die "Checksum mismatch for $(basename "$file")"
      say "Checksum OK"
    fi
  fi
  echo "$file"
}

sudo_cmd() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

# ── .deb path (Debian / Ubuntu / Mint …) ──────────────────────────────────────
if [ "$FORCE_APPIMAGE" = 0 ] && have apt-get && have dpkg && { [ "$(id -u)" -eq 0 ] || have sudo; }; then
  URL="$(asset_url "-${DEB_ARCH}\.deb\$")" || true
  if [ -n "${URL:-}" ]; then
    DEB="$(fetch "$URL")"
    say "Installing with apt (may ask for your password)"
    sudo_cmd apt-get install -y "$DEB"
    say "Installed. Run:  onnxviz model.onnx   (or find “ONNX Viz” in your app launcher)"
    exit 0
  fi
  warn "No .deb found in this release; falling back to the AppImage."
fi

# ── AppImage path (any distro, no root) ───────────────────────────────────────
URL="$(asset_url "-${ARCH}\.AppImage\$")" || true
[ -n "${URL:-}" ] || die "No AppImage found in release $TAG."
IMG="$(fetch "$URL")"
mkdir -p "$BIN_DIR" "$APP_DIR" "$DESKTOP_DIR" "$ICON_DIR"
chmod +x "$IMG"

if [ -e /dev/fuse ] && { ldconfig -p 2>/dev/null | grep -q "libfuse\.so\.2"; }; then
  mv -f "$IMG" "$APP_DIR/onnxviz.AppImage"
  EXEC="$APP_DIR/onnxviz.AppImage"
else
  warn "FUSE is not available; unpacking the AppImage instead (install libfuse2 to avoid this)."
  ( cd "$TMP" && "$IMG" --appimage-extract >/dev/null )
  rm -rf "$APP_DIR/app"; mv "$TMP/squashfs-root" "$APP_DIR/app"
  EXEC="$APP_DIR/app/AppRun"
fi
ln -sf "$EXEC" "$BIN_DIR/onnxviz"

curl -fsSL -o "$ICON_DIR/onnxviz.png" "https://raw.githubusercontent.com/$REPO/$TAG/build/icon.png" || warn "Could not fetch the icon."
cat > "$DESKTOP_DIR/onnxviz.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=ONNX Viz
Comment=Fast, offline ONNX model visualizer
Exec=$EXEC %f
Icon=onnxviz
Terminal=false
Categories=Development;
MimeType=application/x-onnx;
DESK
have update-desktop-database && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true

say "Installed to $APP_DIR"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) warn "$BIN_DIR is not on your PATH — add it, e.g.  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;; esac
say "Run:  onnxviz model.onnx   (or find “ONNX Viz” in your app launcher)"
