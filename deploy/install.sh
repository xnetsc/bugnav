#!/usr/bin/env bash
# Install bugnav on a Debian/Ubuntu host (mini PC, Pi, or cloud server).
# Idempotent. Run as root or with sudo from the repo root.
set -euo pipefail

PREFIX=/opt/bugnav
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> installing to $PREFIX"
mkdir -p "$PREFIX"
cp -r "$SRC_DIR/bugnav" "$PREFIX/"
cp "$SRC_DIR/requirements.txt" "$PREFIX/"
[ -f "$PREFIX/config.json" ] || cp "$SRC_DIR/config.example.json" "$PREFIX/config.json"
mkdir -p "$PREFIX/data"

echo "==> python venv + deps"
python3 -m venv "$PREFIX/.venv"
"$PREFIX/.venv/bin/pip" install -q -r "$PREFIX/requirements.txt"

if [ ! -f "$PREFIX/bugnav.env" ]; then
  echo "==> create $PREFIX/bugnav.env with your Ecovacs password (only place it may live)"
  printf 'ECOVACS_PASSWORD=\n' > "$PREFIX/bugnav.env"
fi
chmod 600 "$PREFIX/bugnav.env"
chmod 600 "$PREFIX/config.json" || true

echo "==> systemd unit"
cp "$SRC_DIR/deploy/bugnav.service" /etc/systemd/system/bugnav.service
systemctl daemon-reload
systemctl enable bugnav

cat <<'EOF'
Done. Next steps:
  1. Edit /opt/bugnav/config.json  (account short ID, country, device_id)
  2. Edit /opt/bugnav/bugnav.env   (ECOVACS_PASSWORD=...)
  3. systemctl start bugnav && journalctl -u bugnav -f
  The daemon commands the robot for real (no shadow modes).
EOF
