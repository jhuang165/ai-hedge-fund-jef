# Deploying the hourly reporter

Two containers, one `docker compose up`:

| service | what | image |
|---|---|---|
| `codex-wrapper` | [circlemouth/Codex-Wrapper](https://github.com/circlemouth/Codex-Wrapper): an OpenAI-compatible HTTP API around the Codex CLI, signed in with your ChatGPT account | built from the pinned upstream commit |
| `reporter` | `aihf reporter` from this repo: evidence → brief per name every hour → SQLite → dashboard | built from `deploy/reporter.Dockerfile` |

The wrapper never gets a host port; only the reporter (port 8788, HTTP basic auth) is reachable.
The model is whatever `codex-home/config.toml` says (`gpt-6-luna`), with Codex's live web search on,
so each brief is written from news the model found itself plus the RSS headlines and the quant
desk's numbers the reporter hands it.

## What you need

- A box that stays on, with Docker. ~1 GB RAM free, ~3 GB disk for the two images.
- `~/.codex/auth.json` from a machine where `codex login` succeeded (your ChatGPT sign-in).
  Codex refreshes the tokens in place, so the file must be writable by UID 1000 in the container.
- Nothing else: no OpenAI API key, no data-vendor key (prices and headlines are Yahoo / Google RSS).

## Run it

```bash
cd deploy
cp .env.example .env            # set PROXY_API_KEY and REPORTER_PASSWORD at minimum
cp ~/.codex/auth.json codex-home/
docker compose up -d --build    # first build: ~5 min (node + codex + python deps)
docker compose logs -f reporter
```

Then open `http://<host>:8788` as user `desk` with `REPORTER_PASSWORD`. The first cycle starts at
boot (`REPORTER_RUN_ON_BOOT`, default on) and takes a few minutes for ten names; afterwards a
cycle runs every `REPORTER_INTERVAL_MINUTES` (60) between `REPORTER_ACTIVE_HOURS` (07:00-20:00
New York) on weekdays and every `REPORTER_OFF_HOURS_INTERVAL_MINUTES` (240) the rest of the time.
Symbols are added and removed on the page (or seeded by `REPORTER_SYMBOLS` on the first boot).

Sanity checks:

```bash
docker compose exec reporter curl -s http://127.0.0.1:8788/healthz
docker compose exec codex-wrapper sh -c 'curl -s -H "Authorization: Bearer $PROXY_API_KEY" http://127.0.0.1:8000/v1/models'
```

The second should list `gpt-6-luna`. If it lists only `codex-cli`/`gpt-5.1`, the wrapper could not
read `codex-home/config.toml` (ownership: `chown -R 1000:1000 codex-home`).

## Where to host it for free

The stack needs an always-on process, a writable disk, and an outbound network. That rules out
the platforms whose free tiers sleep (Render, Hugging Face Spaces) or that have closed their free
tiers to new accounts (Koyeb, Fly, Railway). What is left, in order of preference:

1. **Oracle Cloud "Always Free"** — the only major provider with a permanently free, always-on VM
   with enough memory: up to 4 Arm cores / 24 GB (A1.Flex) or two x86 micro VMs. Needs a card for
   identity verification; the Always Free resources are never billed. Steps:
   1. Sign up at cloud.oracle.com, pick a home region near you.
   2. Compute → Instances → Create: image *Canonical Ubuntu 24.04*, shape *VM.Standard.A1.Flex*
      (1 OCPU / 6 GB is plenty; smaller shapes also dodge the capacity errors big ones hit).
      Add your SSH public key. Note the public IP.
   3. Networking → the instance's VCN → Security Lists → Default → *Add Ingress Rule*:
      source `0.0.0.0/0`, protocol TCP, destination port `8788`.
   4. `ssh ubuntu@<ip>`, then run [`oracle-bootstrap.sh`](oracle-bootstrap.sh) (installs Docker,
      opens the port in the VM's own iptables, adds swap, clones the repo). It prints the two
      remaining steps: `scp` your `auth.json` over, fill in `.env`, `docker compose up -d --build`.
   5. Oracle reclaims Always Free VMs that sit below 20% CPU/memory/network for seven days.
      A 1 OCPU / 6 GB shape running this stack normally stays above that on memory; if you get
      the reclaim notice, the fix Oracle documents is upgrading the account to Pay As You Go,
      which keeps the Always Free resources free.
2. **A student credit** — a `.edu` address qualifies for the GitHub Student Developer Pack
   (DigitalOcean $200 for a year, Azure for Students $100/year with no card). A $4/month
   DigitalOcean droplet or a B1s Azure VM runs this stack for the life of the credit, with the
   same steps as above (the bootstrap script works on any Ubuntu).
3. **Your own Mac, published through a tunnel** — zero sign-ups. `brew install cloudflared`,
   then `bash deploy/mac-tunnel.sh`: it starts the containers, keeps the Mac awake
   (`caffeinate -s`), opens a Cloudflare quick tunnel (no account; the URL is random and changes
   each run) and prints the dashboard URL. `bash deploy/mac-tunnel.sh stop` closes the tunnel.
   Docker Desktop restarts the containers after a reboot; rerun the script for the tunnel.
   A fixed URL needs a free Cloudflare account and a named tunnel, or Tailscale Funnel.

`deploy/` is host-agnostic: anything that runs `docker compose` works.

## The public dashboard on Firebase Hosting (free, fixed URL)

Firebase Hosting serves files only: no Python, no scheduler, no Codex. So the reporter cannot
*run* there, but its dashboard can *live* there. With `REPORTER_PUBLISH_DIR` set, the reporter
exports the site after every cycle (`index.html`, the assets, and the JSON the page would
otherwise fetch from `/api`, under `data/`), then runs `REPORTER_PUBLISH_CMD` in that directory
to upload it. The copy on Firebase is read-only (no *Run now*, no watchlist edits, no sign-in);
the box that runs the reporter keeps the live dashboard. See `hedge_fund/reporter/publish.py`.

This directory holds the Firebase side: [`firebase/firebase.json`](firebase/firebase.json) (serve
`site/`, never cache the data files) and `firebase/.firebaserc` (project `aihf-reporter`, which
serves https://aihf-reporter.web.app). The export itself, `firebase/site/`, is git-ignored.

On a Mac that stays on:

```bash
npm install -g firebase-tools && firebase login     # once
bash deploy/mac-publish.sh            # wrapper container up, keep-awake, reporter on 127.0.0.1:8788,
                                      # export + `firebase deploy --only hosting` after every cycle
bash deploy/mac-publish.sh publish    # export and upload the briefs you have, right now
bash deploy/mac-publish.sh log        # follow the reporter log (~/.hedge-fund/reporter.log)
bash deploy/mac-publish.sh stop
```

The reporter runs from the checkout's venv rather than its container so the `firebase` CLI
signed in on the Mac can upload (the wrapper container publishes port 8020 on localhost for it).
Its database is `~/.hedge-fund/reporter.db`. The status bar on the local dashboard shows the
last publish and its error, if any; a failed upload never blocks the next cycle.

On a Linux box (Oracle, a droplet): install Node and `firebase-tools`, run `firebase login:ci`
on a machine with a browser and put the token in `FIREBASE_TOKEN` (or use a service-account
JSON via `GOOGLE_APPLICATION_CREDENTIALS`), then run the reporter on the host with the same
`REPORTER_PUBLISH_*` variables. Each deploy uploads only files whose content changed.

## Running without Docker (development)

```bash
# wrapper, in its own checkout + venv, with a dedicated CODEX_HOME holding auth.json + config.toml
CODEX_HOME=/path/to/codex-home CODEX_ENV_FILE=wrapper.env uvicorn app.main:app --port 8020

# reporter, from this repo
CODEX_WRAPPER_URL=http://127.0.0.1:8020/v1 CODEX_WRAPPER_API_KEY=<PROXY_API_KEY> \
REPORTER_PASSWORD=secret venv/bin/aihf reporter --port 8788
venv/bin/aihf reporter --once --tickers NVDA     # one cycle, no server (cron-friendly)
```

## Notes

- **Quota.** Every brief is one `codex exec` with web search. Ten names, hourly, 13 hours a day,
  plus four-hourly nights and weekends is roughly 170 runs per weekday. On a ChatGPT Plus plan
  that is likely to hit the Codex usage limit; the symptoms are briefs that fail with an HTTP 429
  or a wrapper timeout, visible per card on the dashboard. Loosen `REPORTER_INTERVAL_MINUTES`,
  shorten `REPORTER_ACTIVE_HOURS`, or set `REPORTER_OFF_HOURS_INTERVAL_MINUTES=0`.
- **Terms.** The wrapper's README notes that OAuth sign-in is for your own use; keep the
  dashboard password-protected and the wrapper unexposed, which this compose file does.
- **Upgrading the wrapper.** The build context is pinned to a commit. To move it, change the SHA
  in `docker-compose.yml`, rebuild, and re-check `/v1/models`.
- **Backups.** Everything the reporter writes is in the `reporter-data` volume
  (`docker compose cp reporter:/data/reporter.db .`).
