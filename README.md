# bugnav — T80s 防卡困外置导航

给科沃斯 T80s 加一个外置“大脑”：卡住时先沿障碍物小幅换向找开口，
找不到就沿来路退出、本次任务放弃该区域。机器人本体零改动，程序跑在
一台常开宿主机上，经科沃斯云端（deebot-client）读位置、发命令。

**完整方案（硬件选型→软件设计→部署安装→操作流程→验证清单→运维回滚）
见 [方案.md](方案.md)。**

快速开始：

```bash
# 1) 先在模拟器里看它怎么工作（不需要真机、不需要账号）
python3 -m bugnav.demo
python3 -m unittest discover -s tests

# 2) 部署到宿主机（默认 observe 只读模式，验证通过才改 active）
cp config.example.json config.json   # 填科沃斯账号短 ID
sudo bash deploy/install.sh          # 或用 docker compose up -d
```

目录：`bugnav/`（核心逻辑+模拟器+真机适配）、`deploy/`（systemd 安装）、
`tests/`（单元与三场景端到端测试）、`Dockerfile`、`docker-compose.yml`。
