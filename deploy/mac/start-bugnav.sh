#!/usr/bin/env bash
# bugnav on Mac (Docker). Run:  bash ~/bugnav/start-bugnav.sh
# - starts Docker Desktop if needed and waits for the engine
# - first run asks for the Ecovacs short ID + password (password is only
#   written to ./.env with 600 perms, never echoed)
# - builds and starts the container in OBSERVE mode (never commands the
#   robot until you switch to active - see 说明.md)
set -euo pipefail
cd "$(dirname "$0")"
echo "== bugnav 容器部署（Mac）=="

if ! command -v docker >/dev/null 2>&1; then
  echo "没找到 docker 命令。请先打开一次 Docker Desktop 完成初始化，然后重新运行本脚本。"
  open -a Docker || true
  exit 1
fi
open -a Docker >/dev/null 2>&1 || true
printf "等待 Docker 引擎就绪"
for _ in $(seq 1 90); do
  if docker info >/dev/null 2>&1; then break; fi
  printf "."
  sleep 2
done
echo
docker info >/dev/null 2>&1 || { echo "Docker 引擎还没起来，等鲸鱼图标稳定后重新运行本脚本。"; exit 1; }

if [ ! -f config.json ]; then
  cp config.template.json config.json
  read -rp "科沃斯账号短 ID（App 个人资料里形如 EXXXXXXX，不要填手机号）: " ACC
  sed -i '' "s/__ACCOUNT_ID__/${ACC}/" config.json
  chmod 600 config.json
  echo "已生成 config.json"
fi

if [ ! -s .env ]; then
  read -rsp "科沃斯账号密码（输入时不显示，输完回车）: " PW
  echo
  printf "ECOVACS_PASSWORD='%s'\n" "$PW" > .env
  chmod 600 .env
  echo "已写入 .env（仅本机，权限 600）"
fi

mkdir -p data
docker compose up -d --build
echo
echo "== 容器已启动（observe 只读模式）。最近日志： =="
sleep 3
docker compose logs --tail 40 bugnav || true
echo
echo "看实时日志:  docker compose -f ~/bugnav/docker-compose.yml logs -f bugnav"
echo "停止:        docker compose -f ~/bugnav/docker-compose.yml down"
