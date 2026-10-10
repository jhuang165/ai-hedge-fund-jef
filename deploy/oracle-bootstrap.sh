#!/usr/bin/env bash
# One-shot setup of an Oracle Cloud "Always Free" Ubuntu VM (ARM A1.Flex or x86
# E2.1.Micro) for the reporter stack. Run as the default `ubuntu` user:
#
#   bash oracle-bootstrap.sh
#
# Then copy auth.json into ~/ai-hedge-fund-jef/deploy/codex-home/, write
# deploy/.env, and `docker compose up -d --build`. See deploy/README.md.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jhuang165/ai-hedge-fund-jef.git}"
BRANCH="${BRANCH:-main}"
PORT="${REPORTER_PORT:-8788}"

echo "== packages"
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl git

echo "== docker (official repository)"
if ! command -v docker >/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER"
fi

echo "== firewall: Oracle's Ubuntu image drops everything but port 22 in iptables"
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$PORT" -j ACCEPT || true
if ! dpkg -s iptables-persistent >/dev/null 2>&1; then
  echo iptables-persistent iptables-persistent/autosave_v4 boolean true | sudo debconf-set-selections
  echo iptables-persistent iptables-persistent/autosave_v6 boolean true | sudo debconf-set-selections
  sudo apt-get install -y iptables-persistent
fi
sudo netfilter-persistent save || true
echo "   (also add an ingress rule for TCP $PORT in the VCN security list in the Oracle console)"

echo "== swap (the free shapes have none; pandas imports spike memory)"
if ! swapon --show | grep -q .; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== repository"
if [ ! -d "$HOME/ai-hedge-fund-jef" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$HOME/ai-hedge-fund-jef"
fi
cd "$HOME/ai-hedge-fund-jef/deploy"
[ -f .env ] || cp .env.example .env
mkdir -p codex-home
sudo chown -R 1000:1000 codex-home

cat <<MSG

Done. Next, from the machine where you ran \`codex login\`:

  scp ~/.codex/auth.json ubuntu@<vm-ip>:~/ai-hedge-fund-jef/deploy/codex-home/auth.json

then on the VM:

  cd ~/ai-hedge-fund-jef/deploy
  nano .env                       # PROXY_API_KEY and REPORTER_PASSWORD at minimum
  newgrp docker                   # once, or log out and back in
  docker compose up -d --build
  docker compose logs -f reporter

Dashboard: http://<vm-ip>:$PORT  (user "desk", your REPORTER_PASSWORD)
MSG
