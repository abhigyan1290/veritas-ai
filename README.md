# Veritas

**AI cost observability for developers.** Track token usage, costs, and latency across every LLM call — automatically. Spot regressions before they hit production.

---

## Beta Install

```bash
pip install git+https://github.com/abhigyan1290/veritas-ai.git
```

To update to the latest version:
```bash
pip install --upgrade git+https://github.com/abhigyan1290/veritas-ai.git
```

---

## Quickstart

You'll receive two things from us: a **username/password** to log into the dashboard, and an **API key** for your project.

### 1. Log into the Dashboard

Go to **https://web-production-82424.up.railway.app/auth/login** and sign in with the credentials we gave you.

Create a new project to get your API key — it looks like `sk-vrt-...`. **Copy it immediately**, it's only shown once.

### 2. Add Veritas to your code

It's a single `init()` call plus wrapping your existing client:

```python
import veritas
import anthropic

# Configure once at startup
veritas.init(
    api_key="sk-vrt-YOUR-KEY-HERE",
    endpoint="https://web-production-82424.up.railway.app/api/v1/events"
)

# Wrap your existing client — one line change
client = veritas.Anthropic(anthropic.Anthropic(), feature_name="my_feature")

# Use exactly as before — nothing else changes
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

That's it. Every API call is now tracked in your dashboard.

---

## OpenAI Support

```python
import veritas
import openai

veritas.init(
    api_key="sk-vrt-YOUR-KEY-HERE",
    endpoint="https://web-production-82424.up.railway.app/api/v1/events"
)

client = veritas.OpenAI(openai.OpenAI(), feature_name="my_feature")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Environment Variable Config (Alternative)

Instead of calling `veritas.init()`, you can set environment variables:

```bash
VERITAS_API_KEY=sk-vrt-YOUR-KEY-HERE
VERITAS_API_URL=https://web-production-82424.up.railway.app/api/v1/events
```

Veritas auto-configures on import if these are set.

---

## What Gets Tracked

For every LLM call, Veritas automatically captures:

| Field | Description |
|---|---|
| `feature` | The name you pass to `feature_name=` |
| `model` | The model used (e.g. `claude-3-haiku`) |
| `tokens_in` / `tokens_out` | Input and output token counts |
| `cost_usd` | Computed cost based on current pricing |
| `latency_ms` | End-to-end request time |
| `code_version` | Current git commit hash (auto-detected) |

---

## Safety Guarantees

- Veritas **never crashes your application** — all tracking is fire-and-forget
- No data is stored locally by default (everything goes to the dashboard)
- Your actual LLM responses are never transmitted — only metadata

---

## Beta Feedback

We're in closed beta. Please share any issues or feedback directly with us.
