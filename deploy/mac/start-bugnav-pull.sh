#!/usr/bin/env bash
# bugnav on Mac, pull variant: no code on the Mac, image comes from GHCR.
# Run:  bash ~/bugnav/start-bugnav.sh
set -euo pipefail
cd "$(dirname "$0")"
echo "== bugnav 容器启动（拉取镜像版）=="

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
  chmod 600 config.json
  echo "已生成 config.json（不含任何凭据）"
fi

# Credentials go ONLY into ./.env (600) as container environment
# variables. They are never written into config.json or any code file.
# Append-only: an existing .env (e.g. with LLM keys) is kept; the
# Ecovacs lines are added only when missing.
touch .env
if ! grep -q '^ECOVACS_ACCOUNT_ID=' .env; then
  read -rp "科沃斯账号短 ID（App 个人资料里形如 EXXXXXXX，不要填手机号）: " ACC
  read -rsp "科沃斯账号密码（输入时不显示，输完回车）: " PW
  echo
  [ -n "$(tail -c 1 .env)" ] && echo >> .env
  printf "ECOVACS_ACCOUNT_ID='%s'\nECOVACS_PASSWORD='%s'\n" "$ACC" "$PW" >> .env
  chmod 600 .env
  echo "凭据已写入 .env（容器环境变量，仅本机，权限 600）"
fi
chmod 600 .env 2>/dev/null || true

mkdir -p data
docker compose pull
docker compose up -d
echo
echo "== 容器已启动。最近日志： =="
sleep 3
docker compose logs --tail 40 bugnav || true
echo
echo "看实时日志:  docker compose -f ~/bugnav/docker-compose.yml logs -f bugnav"
echo "停止:        docker compose -f ~/bugnav/docker-compose.yml down"
