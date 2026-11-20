# LearnHarness + this stack — working manual

A clone of [learnharness](https://github.com/amosblomqvist/learnharness) now lives at
`./learnharness/` (inside this workspace). This manual covers how to run it **and** how to
point it at the local Qwen model server this repo already serves, so LearnHarness does its
teaching against *your* model instead of a paid cloud provider.

- **What it is:** an "agent harness for learning difficult things" — a Next.js + Electron
  desktop/web app that maps what you know to a goal and teaches step by step, keeping a
  persistent knowledge graph. TypeScript, **AGPL-3.0**.
- **Where it is:** `./learnharness/` — git-ignored on purpose (see
  [Keeping the clone clean](#5-keeping-the-clone-clean)). It is a *separate* upstream repo
  with its own `origin`; it is not part of this inference repo's history.
- **Why the clone is ignored:** it matches this repo's existing "nested clones" convention
  in `.gitignore`. Do not `git add learnharness/` into this repo.

---

## 1. Prerequisites

- **Node + npm.** Reuse the Node this stack already pins — no need for a second install:
  ```bash
  export PATH="$HOME/.nvm/versions/node/v22.22.3/bin:$PATH"
  node -v   # v22.22.3
  ```
  (That's the `NODE_BIN` from `scripts/dsh-env.sh`. Any current Node LTS works; this one is
  already on your machine.)
- **A model provider.** For "your work," that's this repo's local server (section 4). No
  cloud API key is required for the custom-endpoint path.
- LearnHarness stores chats, graph, settings, and keys **locally**. It does not download
  models or install a backend.

---

## 2. First run (web dev mode)

```bash
cd learnharness
npm ci
npm run dev
```

Open **http://localhost:3000**. (No clash with this repo's own web UI on `:3090`, or the
model ports `:8000`/`:8080`/`:8001`.)

Other useful scripts (from its `package.json`):

```bash
npm test        # tests against fixtures + fake providers
npm run lint
npm run typecheck
npm run desktop:dev   # runs Next + Electron together (the real desktop shell)
```

---

## 3. First run (desktop app)

```bash
cd learnharness
npm ci
npm run desktop:dev
```

This starts `next dev` and an Electron window pointed at it. Use this when you want the
real desktop experience (the `desktop:dist` build target is for signed/notarized releases
and needs Xcode + Apple notarization creds — not needed for local work).

---

## 4. Wire it to **your** local model (the important part)

LearnHarness supports **one custom OpenAI-compatible endpoint**, and this repo serves
exactly that at `/v1`. So you teach against your own Qwen build.

### 4a. Start a model server from this repo

Pick the backend you want LearnHarness to use (run from the repo root, with the stack env
sourced — `source scripts/dsh-env.sh` or your usual launcher):

| Backend | Start it with | Serves OpenAI API at | Default model id |
|---|---|---|---|
| **Splash** (default) | `scripts/serve-splash.sh` | `http://127.0.0.1:8000/v1` | `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` |
| **llama.cpp** | `scripts/serve-llama.sh` | `http://127.0.0.1:8080/v1` | `qwen3.8-27b-local` |
| **lithos** | `scripts/serve-lithos.sh` | `http://127.0.0.1:8001/v1` | (from `/v1/models`) |

Confirm it's up and get the **exact** model id the server reports (don't guess it — let
discovery or this curl tell you):

```bash
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

(Swap the port for `:8080` or `:8001` if you started llama/lithos.)

### 4b. Add it as a Custom endpoint in LearnHarness

1. **Settings → Providers → + Add a provider → Custom endpoint** (or the first-run picker).
2. **Base URL:** paste the full URL **including `/v1`** — LearnHarness does *not* append it:
   - Splash → `http://127.0.0.1:8000/v1`
   - llama  → `http://127.0.0.1:8080/v1`
   - lithos → `http://127.0.0.1:8001/v1`
3. **Auth:** choose **Keyless** (this stack's local server is keyless). If you'd rather send
   a key, choose **API key** and enter `sk-local` — the placeholder `LLAMA_API_KEY` the
   server accepts.
4. **Model IDs:** leave blank to let **Connect** discover them via `GET <base>/models`. If
   discovery comes back empty, type the id from the curl above (e.g.
   `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` for Splash, or `qwen3.8-27b-local` for llama).
5. **Connect**, then pick it as the **Strong** / **Cheap** / **Vision** model.

### 4c. Match the context budget to your server

LearnHarness's per-model **Context budget** and **Max output** are *request/compaction*
budgets — they do **not** resize your backend. Set them to match what you actually serve:

- This stack's `CTX` default is **131072**. Pin **Context budget = 131072** (or lower, e.g.
  `65536`, if you launched with `SPLASH_EXTRA_ARGS=--max-context 64K`).
- Leave **Max output** at the default unless you have a reason to cap it.

If you don't pin, LearnHarness falls back to **8192 / 4096** — far below what your server can
do — so pinning is worth it.

### 4d. (Optional) Claude Code CLI as a separate provider

LearnHarness can also use an installed **Claude Code CLI** as a provider
(**Settings → Providers → External CLI → Claude Code**, min version `2.1.259`). This is
*separate* from your local model and uses the CLI's own auth/billing — not your Qwen
server. Use the custom endpoint above for local-model teaching; reach for this only if you
specifically want Claude models inside LearnHarness.

---

## 5. Keeping the clone clean

The clone is its own repo. Keep its changes out of *this* repo and vice versa:

- **Update it:**
  ```bash
  cd learnharness && git pull
  ```
- It's already in this repo's `.gitignore` as `learnharness/`, so `git status` here stays
  clean. Don't `git add` it.
- **`next dev` rewrites `AGENTS.md`** in the clone each run (an upstream behavior). That's an
  expected uncommitted change *inside the clone*, not something this repo tracks.

---

## 6. Gotchas

- **AGPL-3.0.** If you distribute a modified LearnHarness (including over a network), the
  license's source-offer terms apply. Fine for local/personal use; mind it before shipping.
- **"This is NOT the Next.js you know."** The clone's `AGENTS.md` warns that its Next.js has
  breaking changes vs. training data — read `node_modules/next/dist/docs/` before writing
  app code there, and don't guess design/behavior (ask).
- **Discovery limits:** connect times out at 15s, caps at 8 MiB / 10,000 models. A 401/403,
  redirect, or oversize response fails the connect even if you typed model IDs.
- **"Off" may not stop thinking.** The reasoning-effort "Off" setting declares a capability;
  it may not disable a backend's default thinking. Configure reasoning on the server side if
  you need it off.
- **Pricing shows as unknown, not $0** — even for your local server. That's expected; there's
  no rate editor.

---

## 7. Quick reference

```bash
# One-time
export PATH="$HOME/.nvm/versions/node/v22.22.3/bin:$PATH"
cd learnharness && npm ci

# Every session: 1) serve your model (repo root), 2) run LearnHarness
scripts/serve-splash.sh                      # -> http://127.0.0.1:8000/v1
cd learnharness && npm run dev               # -> http://localhost:3000
# then: Settings -> Providers -> Custom endpoint -> http://127.0.0.1:8000/v1 (Keyless)

# Verify the server + model id
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool

# Keep the clone current
cd learnharness && git pull
```

| Backend | OpenAI base URL | Auth | Context pin |
|---|---|---|---|
| Splash (default) | `http://127.0.0.1:8000/v1` | Keyless (or `sk-local`) | `131072` |
| llama.cpp | `http://127.0.0.1:8080/v1` | Keyless (or `sk-local`) | `131072` |
| lithos | `http://127.0.0.1:8001/v1` | Keyless (or `sk-local`) | match `--max-context` |
