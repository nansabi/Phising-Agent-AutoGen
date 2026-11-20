# phishing-desk

## Project overview

`phishing-desk` is a defensive email-analysis demo. It extracts email facts,
checks sender and link indicators with offline rules, and uses AutoGen agents
to classify and debate a verdict. Later phases add approval, case storage,
policy lookup, logging, reports, and a risk chart. Bundled samples are
synthetic and use reserved example addresses and domains.

## Install

Use Python 3.12 and run these commands from the project directory in
PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`; it is required for online analysis.
`OPENAI_API_KEY` is optional and enables the fallback model. Never commit
`.env`. Docker Desktop is needed for Phase 6B's default chart executor. Use
`--no-docker` to explicitly opt into the local chart fallback. The Docker
Matplotlib setup uses a 120-second pip timeout and retries downloads three
times.

## Run a feature

Run commands from this directory. The CLIs accept `--text`, `--file`, or
`--image`; with no input option, they prompt for pasted email text ending with
`END`.

| Entry point | Feature |
| --- | --- |
| `python main_phase1.py` | Two-agent fact extraction and classification |
| `python main_phase2.py` | Sequential extraction, link checks, and classification |
| `python main_phase3.py` | Classifier and Skeptic group-chat debate |
| `python main_phase4.py` | Desk_Chief nested-chat summary (`--verbose` shows internal steps) |
| `python main_phase5.py` | Human approval, notes, and save/resume cases |
| `python main_phase6.py` | Policy lookup, SQLite logs, and reports |
| `python main_phase6b.py` | **Full version:** all analysis steps, policy, reports, Docker chart, cache demo, and optional offline mode |
| `uvicorn api.main:app --reload --port 8000` | Phase 7A local FastAPI backend |

Example commands:

```powershell
python main_phase6b.py --file samples\phish1.txt
python main_phase6b.py --file samples\phish1.txt --offline --no-docker
python main_phase5.py --list-cases
python main_phase6b.py --show-log --log-limit 5
```

The sample email checks are rule-based and make no network or WHOIS requests.
Screenshot transcription sends image bytes to the configured vision model;
links extracted from an image come from visible text only and may not match
the actual URLs. Saved cases, reports, charts, logs, local inputs, and cache
data remain in git-ignored folders.

## Phase 7B - React web interface

The React and TypeScript interface in `web/` uses the Phase 7A API at
`http://127.0.0.1:8000`. Start the backend in one terminal from the project
root, then start the Vite website in another terminal:

```powershell
# Terminal 1, project root
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --reload --port 8000

# Terminal 2
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. The Analyze page accepts pasted text or a
`.txt`, `.eml`, or image upload. Cases include live progress, evidence,
approval actions, and the generated report; Cases and Logs & Costs provide
saved-case and observability views. API-provided links are shown in defanged
form and email content is rendered as plain text. Run the frontend checks
from `web/` with `npm run test` and `npm run build`.
