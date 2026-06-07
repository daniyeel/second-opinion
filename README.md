<div align="center">
  <img src="public/logo.png" alt="Second Opinion" width="72" height="72" />
  <h1>Second Opinion</h1>
  <p><strong>Real-time AI clinical reasoning. Transparent, auditable, educational.</strong></p>

  <p>
    <a href="https://second-opinion-
      .up.railway.app"><img src="https://img.shields.io/badge/Live%20Demo-Railway-6366f1?style=flat-square&logo=railway" alt="Live demo" /></a>
    <img src="https://img.shields.io/badge/Node.js-18%2B-339933?style=flat-square&logo=node.js" alt="Node.js" />
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License" />
    <img src="https://img.shields.io/badge/Models-OpenRouter-ff6b35?style=flat-square" alt="OpenRouter" />
  </p>

  <p>
    <a href="https://second-opinion.up.railway.app">View Demo</a> ·
    <a href="#getting-started">Get Started</a> ·
    <a href="#architecture">How It Works</a>
  </p>
</div>

---

## The problem

Most AI tools in medicine return a diagnosis. Second Opinion returns the *reasoning process behind it*.

Clinical reasoning is the hardest skill to teach. A diagnosis without explanation is useless to a student and indefensible to a clinician. When an experienced physician evaluates a patient, they don't jump to a conclusion: they build a list of possibilities, weigh the evidence, revise their thinking in real time, and ask the right questions. That process has never been made visible. Until now.

## What it does

Second Opinion takes a free-text patient presentation and streams a live, structured differential diagnosis: not as a chat response, but as a typed reasoning trace with auditable steps.

| Feature | Description |
|---|---|
| **Live differential** | Top-3 hypotheses with animated probability bars, updated in real time as the model reasons |
| **Evidence** | Supporting and refuting clinical findings surfaced per hypothesis |
| **Clinical summary** | 3-4 sentence conclusion: leading diagnosis justified by specific clinical findings, why the runner-up is less likely, then next steps and urgency |
| **Clarifying questions** | 2-3 targeted questions that would most narrow the differential |
| **Refinement** | Answer the questions or add new findings - each pass shown as a labeled block in the input card, differential reruns with updated context |
| **Comparison** | Run two models on the same case simultaneously, side by side; each pane saves to history independently |
| **Multi-Agent (CAMEL-AI)** | Three specialist agents - Internist, domain-adaptive Specialist, and Devil's Advocate - each independently analyze the case using the user's selected model, then discuss, then synthesize a consensus differential. If a normal run was done first, the agents receive that differential as context and review it as a genuine second opinion |
| **Guided form** | "Fill with form" builds a structured patient description from fields (age, sex, chief complaint, symptoms, duration, vitals, history) and compiles it into the presentation textarea |
| **PDF export** | Clinical-quality report: leading diagnosis with probability, differential with evidence, refinement passes, full reasoning log, disclaimer footer on every page. Multi-agent runs additionally include each specialist's individual differential, the inter-agent discussion transcript, and the prior assessment context if present |
| **Session history** | Last 10 runs saved locally - including compare runs - restorable in one click, deletable per entry |
| **Dark mode** | Full dark/light theme with OS preference detection |
| **Multilingual** | Complete Simplified Chinese interface, including all model output |

Every step is visible as it happens. The model cannot output a bare answer; it must justify each decision, one structured event at a time.

---

## Architecture

### Structured output over SSE

The system prompt instructs the model to respond exclusively in **NDJSON**: one typed JSON object per line, nothing else. Seven event types are defined: `thought`, `hypothesis`, `update`, `evidence`, `tests`, `conclusion`, and `question`. The server streams these to the client over **Server-Sent Events**. The frontend parses each line the moment it arrives; the live differential updates, probability bars animate, and evidence tags attach - all in real time.

### Prompt engineering as a schema enforcer

The model is not asked to "think about" a diagnosis. It is given a strict output contract with required field names, sequencing rules, and cardinality constraints. Malformed lines are silently discarded server-side before reaching the client. This turns an open-ended language model into a structured reasoning engine.

### Model comparison

Two models stream in parallel via `Promise.all`, each building its own differential independently. When both complete, results sit side by side: same patient, same moment, different reasoning paths. Each pane has its own PDF export and saves to session history.

### Multi-agent reasoning (CAMEL-AI)

A Python FastAPI microservice runs three specialist agents sequentially: an Internist (broad differential), a domain-adaptive Specialist (Cardiologist, Neurologist, etc. - inferred from the presentation), and a Devil's Advocate (challenges anchoring, surfaces rare diagnoses). Each agent uses whichever model the user selected in the UI. Each outputs structured NDJSON into its own pane. A discussion moderator then facilitates a brief inter-agent exchange. Finally, a synthesis agent produces a consensus differential and a full 3-4 sentence clinical summary.

If a normal single-model run already completed in the session, the specialist agents receive that differential as context - making the multi-agent pass a genuine second opinion rather than a fresh cold start. A "Convene specialists" button appears next to "Save PDF" after any normal run to trigger this flow directly.

The Node.js server proxies SSE from the Python service on `/api/camel-reason`. Both services share the same `OPENROUTER_API_KEY`; no additional credentials are needed.

### Iterative refinement

After the conclusion, the user can answer the model's clarifying questions or add new clinical detail. The original presentation stays unchanged in the textarea; each refinement pass appears below it as a labeled block with an accent border. The full context (original + all passes) is sent to the model for a new reasoning run.

### PDF export

`/api/pdf` receives the rendered report HTML, spins up a headless Chromium instance via Puppeteer, and calls `page.pdf()`. The report includes the leading diagnosis with probability and confidence, the top-3 differential with evidence, any refinement passes, and the full reasoning log. For multi-agent runs, the report additionally includes each specialist's top differential, the inter-agent discussion transcript, and the prior assessment context if present. A disclaimer footer is pinned to the bottom of every page. Text is fully selectable.

---

## Getting started

### Prerequisites

- Node.js 18+
- Python 3.10+ (for Multi-Agent mode)
- An [OpenRouter](https://openrouter.ai) API key

### Installation

```bash
git clone https://github.com/daniyeel/second-opinion.git
cd second-opinion
npm install
pip install -r camel-service/requirements.txt
```

### Configuration

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

### Run

**Standard mode** (single-model reasoning, compare, guided form):
```bash
npm start
```

**With Multi-Agent** (requires the Python service in a second terminal):
```bash
npm run camel
```

Open [http://localhost:3000](http://localhost:3000). Multi-Agent is available as soon as both processes are running. If only the Node server is running, clicking "Convene specialists" returns a graceful error.

---

## Tech stack

| Layer | Technology |
|---|---|
| Runtime | Node.js 18+ |
| Server | Express |
| Streaming | Server-Sent Events (SSE) |
| AI gateway | OpenRouter |
| Multi-agent service | Python 3.10+ · FastAPI · OpenAI SDK |
| PDF rendering | Puppeteer (headless Chromium) |
| Frontend | Vanilla JS (no framework, no build step) |
| Hosting | Railway |

## Models

| Model | ID |
|---|---|
| MiniMax M2 | `minimax/minimax-m2` |
| Gemini 2.0 Flash | `google/gemini-2.0-flash-001` |
| GPT-4o Mini | `openai/gpt-4o-mini` |
| GPT-4o | `openai/gpt-4o` |
| Claude Sonnet 4.6 | `anthropic/claude-sonnet-4-6` |

All models are routed through OpenRouter. MiniMax M2 is the default. Any model on OpenRouter can be swapped in with a one-line change in `server.js`.

---

## Project structure

```
second-opinion/
├── server.js               # Express server - SSE streaming, PDF endpoint, CAMEL proxy
├── camel-service/
│   ├── main.py             # FastAPI multi-agent service - three specialist agents + synthesis
│   └── requirements.txt    # Python deps: fastapi, uvicorn, openai, python-dotenv
├── public/
│   ├── index.html          # HTML skeleton and theme-init script
│   ├── style.css           # All styles - design tokens, components, dark mode, responsive
│   ├── app.js              # All client logic - i18n, streaming, rendering, multi-agent, PDF
│   └── logo.png            # Brand mark
├── package.json
└── .env                    # OPENROUTER_API_KEY (not committed)
```

No build step. No bundler. No framework.

## License

MIT

---

<div align="center">
  <sub>Educational use only. Not for clinical or diagnostic purposes.</sub>
</div>
