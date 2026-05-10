import re
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


APP_TITLE = "Reflection Agent - Ollama"
PROVIDER = "Ollama"
MODEL = "qwen2.5:0.5b"
OLLAMA_URL = "http://localhost:11434/api/chat"

app = FastAPI(title=APP_TITLE, version="1.0")


GENERATOR_PROMPT = """
You are an expert Python developer.
When given a task, write a clean, working Python function.
Include a docstring.
Handle edge cases.
Return ONLY the Python code - no explanation, no markdown fences.
""".strip()

CRITIC_PROMPT = """
You are a senior Python code reviewer.
Evaluate the given code against these five criteria:

1. Correctness - does it solve the exact requested task and produce the right output?
2. Edge cases - does it handle None, empty, zero, negatives?
3. Readability - clear names, comments, easy to follow?
4. Efficiency - no unnecessary loops or operations?
5. Security - no eval on user input, no hardcoded secrets?

For each criterion found to have a problem, write:
ISSUE [criterion]: <specific problem and why it matters>

If ALL five criteria are met with no issues, respond with exactly one word:
APPROVED

Never say APPROVED if any criterion has an issue.
If there are issues, do not include the word APPROVED anywhere in your response.
Do not rewrite the code in your critique.
Report at most one clear issue per criterion.
Do not invent edge cases that are already handled by the code.
Be specific. Generic feedback is useless.
""".strip()

REVISION_PROMPT = """
You are an expert Python developer revising your previous code.

Your original code:
{original}

Original task:
{task}

Code review critique received:
{critique}

Rewrite the function to fix every issue raised.
Do not just acknowledge the critique - actually fix each problem.
Return ONLY the corrected Python code.
""".strip()


class ReflectRequest(BaseModel):
    task: str = Field(..., min_length=1)
    max_rounds: int = Field(default=3, ge=1, le=5)


class ReflectResponse(BaseModel):
    final_code: str
    round_count: int
    approved: bool
    critiques: List[str]


def clean_code_output(text: str) -> str:
    """Remove markdown fences if a model adds them despite the prompt."""
    cleaned = text.strip()
    match = re.match(r"^```(?:python)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned


def normalize_critique(text: str) -> str:
    """Keep weak-model critique output compatible with the exact APPROVED stop rule."""
    cleaned = text.strip()
    upper = cleaned.upper()

    if upper == "APPROVED":
        return "APPROVED"
    if "ISSUE" not in upper and upper.startswith("APPROVED"):
        return "APPROVED"
    if "ISSUE" not in upper and "NO ISSUES" in upper:
        return "APPROVED"

    if "ISSUE" in upper and "APPROVED" in upper:
        filtered_lines = []
        for line in cleaned.splitlines():
            line_upper = line.strip().upper()
            if line_upper == "APPROVED":
                continue
            if "IF ALL FIVE CRITERIA" in line_upper:
                continue
            if "RESPOND WITH EXACTLY" in line_upper:
                continue
            filtered_lines.append(line)
        return "\n".join(filtered_lines).strip()

    return cleaned


def call_llm(system: str, user: str) -> str:
    """
    Same interface as the Groq version.
    Calls Ollama local API instead of Groq cloud API.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def run_reflection(task: str, max_rounds: int) -> ReflectResponse:
    """
    Reflection Pattern:
    1. Generator writes code.
    2. Critic reviews it against a checklist.
    3. If the Critic says APPROVED, stop.
    4. Otherwise feed critique back to the Generator.
    5. Stop after max_rounds to avoid infinite loops.
    """
    critiques: List[str] = []
    current_code = clean_code_output(
        call_llm(
            system=GENERATOR_PROMPT,
            user=f"Write a Python function for this task:\n{task}",
        )
    )

    for round_num in range(1, max_rounds + 1):
        critique = normalize_critique(call_llm(
            system=CRITIC_PROMPT,
            user=f"Original task:\n{task}\n\nReview whether this Python code solves that exact task:\n\n{current_code}",
        ))

        if critique.strip().upper() == "APPROVED":
            critiques.append("APPROVED")
            return ReflectResponse(
                final_code=current_code,
                round_count=round_num,
                approved=True,
                critiques=critiques,
            )

        critiques.append(critique)
        current_code = clean_code_output(
            call_llm(
                system=GENERATOR_PROMPT,
                user=REVISION_PROMPT.format(task=task, original=current_code, critique=critique),
            )
        )

    return ReflectResponse(
        final_code=current_code,
        round_count=max_rounds,
        approved=False,
        critiques=critiques,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "provider": PROVIDER, "model": MODEL}


@app.post("/reflect", response_model=ReflectResponse)
def reflect(req: ReflectRequest) -> ReflectResponse:
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task field cannot be empty")
    try:
        return run_reflection(req.task.strip(), req.max_rounds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML.replace("__PROVIDER__", PROVIDER).replace("__MODEL__", MODEL)


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>M Abdullah Fawad - Reflection Agent</title>
  <style>
    :root {
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #11110f;
      --muted: #686660;
      --line: #dedbd2;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --soft: #eef6f4;
      --danger: #b42318;
      --shadow: 0 18px 50px rgba(28, 27, 23, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .shell {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 26px 0 40px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 14px 0 24px;
      border-bottom: 1px solid var(--line);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }
    .mark {
      width: 40px;
      height: 40px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      color: white;
      background: #11110f;
      font-weight: 800;
      letter-spacing: 0;
      flex: 0 0 auto;
    }
    h1 {
      margin: 0;
      font-size: clamp(22px, 3vw, 34px);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .subtitle {
      margin-top: 5px;
      color: var(--muted);
      font-size: 14px;
    }
    .status {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 11px;
      font-size: 13px;
      color: var(--muted);
      background: rgba(255,255,255,0.68);
      white-space: nowrap;
    }
    main {
      display: grid;
      grid-template-columns: minmax(310px, 430px) minmax(0, 1fr);
      gap: 20px;
      padding-top: 24px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .composer { padding: 18px; }
    label {
      display: block;
      font-weight: 700;
      font-size: 13px;
      margin-bottom: 8px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfbf8;
      color: var(--ink);
      font: inherit;
      outline: none;
      transition: border-color .15s, box-shadow .15s, background .15s;
    }
    textarea {
      min-height: 178px;
      resize: vertical;
      padding: 12px;
      line-height: 1.5;
    }
    input { padding: 11px 12px; }
    textarea:focus, input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 4px rgba(15,118,110,0.12);
      background: #fff;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: end;
      margin-top: 14px;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      height: 44px;
      padding: 0 16px;
      font-weight: 800;
      cursor: pointer;
      min-width: 132px;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: .65; cursor: wait; }
    .examples {
      display: grid;
      gap: 8px;
      margin-top: 14px;
    }
    .example {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfbf8;
      padding: 10px;
      text-align: left;
      color: var(--ink);
      height: auto;
      min-width: 0;
      font-weight: 600;
    }
    .results {
      min-height: 610px;
      overflow: hidden;
    }
    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
    }
    h2 {
      margin: 0;
      font-size: 17px;
      letter-spacing: 0;
    }
    .approval {
      border-radius: 999px;
      padding: 7px 10px;
      font-weight: 800;
      font-size: 12px;
      color: var(--muted);
      background: #f1efe8;
      border: 1px solid var(--line);
      white-space: nowrap;
    }
    .approval.good {
      color: var(--accent-dark);
      background: var(--soft);
      border-color: #b9ddd6;
    }
    .approval.bad {
      color: var(--danger);
      background: #fff1ef;
      border-color: #ffd0ca;
    }
    .last-task {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      max-width: 620px;
    }
    .code {
      margin: 0;
      padding: 18px;
      min-height: 270px;
      overflow: auto;
      border-bottom: 1px solid var(--line);
      background: #10100e;
      color: #f3f1e8;
      font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      white-space: pre-wrap;
    }
    .trace {
      padding: 16px 18px 18px;
      display: grid;
      gap: 12px;
    }
    .round {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfbf8;
    }
    .round strong {
      display: block;
      font-size: 13px;
      margin-bottom: 6px;
    }
    .round p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .empty {
      color: var(--muted);
      padding: 18px;
      line-height: 1.6;
    }
    @media (max-width: 840px) {
      header { align-items: flex-start; flex-direction: column; }
      .status { justify-content: flex-start; }
      main { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <div class="mark">AF</div>
        <div>
          <h1>M Abdullah Fawad</h1>
          <div class="subtitle">Reflection Pattern Agent</div>
        </div>
      </div>
      <div class="status">
        <span class="pill">Provider: __PROVIDER__</span>
        <span class="pill">Model: __MODEL__</span>
      </div>
    </header>

    <main>
      <section class="panel composer">
        <label for="task">Programming task</label>
        <textarea id="task">Write a Python function that checks if a string is a palindrome.</textarea>
        <div class="row">
          <div>
            <label for="rounds">Max rounds</label>
            <input id="rounds" type="number" min="1" max="5" value="3" />
          </div>
          <button id="run">Run</button>
        </div>
        <div class="examples">
          <button class="example" data-task="Write a Python function that checks if a string is a palindrome.">Palindrome checker</button>
          <button class="example" data-task="Write a Python function that takes a list of integers and returns the two indices whose values sum to a given target. Raise a ValueError if no solution exists.">Two-sum indices</button>
          <button class="example" data-task="Write a Python function that normalizes Unicode text and returns a safe word count.">NLP text normalization</button>
        </div>
      </section>

      <section class="panel results">
        <div class="result-head">
          <div>
            <h2>Final output</h2>
            <div id="lastTask" class="last-task">No run yet.</div>
          </div>
          <span id="approval" class="approval">Idle</span>
        </div>
        <pre id="code" class="code">Run a task to generate reviewed Python code.</pre>
        <div id="trace" class="trace">
          <div class="empty">The critique trace appears here after each reflection round.</div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const task = document.querySelector("#task");
    const rounds = document.querySelector("#rounds");
    const run = document.querySelector("#run");
    const code = document.querySelector("#code");
    const trace = document.querySelector("#trace");
    const approval = document.querySelector("#approval");
    const lastTask = document.querySelector("#lastTask");

    document.querySelectorAll(".example").forEach((button) => {
      button.addEventListener("click", () => {
        task.value = button.dataset.task;
        task.focus();
      });
    });

    function setApproval(text, state) {
      approval.textContent = text;
      approval.className = "approval" + (state ? ` ${state}` : "");
    }

    function renderTrace(critiques) {
      if (!critiques.length) {
        trace.innerHTML = '<div class="empty">No critique rounds returned.</div>';
        return;
      }
      trace.innerHTML = critiques.map((critique, index) => `
        <div class="round">
          <strong>Round ${index + 1}</strong>
          <p>${critique.replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}</p>
        </div>
      `).join("");
    }

    run.addEventListener("click", async () => {
      run.disabled = true;
      const runTask = task.value;
      document.querySelectorAll(".example").forEach((button) => button.disabled = true);
      code.textContent = "Generating, reviewing, and revising...";
      trace.innerHTML = '<div class="empty">Waiting for critique rounds.</div>';
      setApproval("Running", "");
      lastTask.textContent = `Last run task: ${runTask}`;

      try {
        const response = await fetch("/reflect", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            task: runTask,
            max_rounds: Number(rounds.value || 3)
          })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Request failed");
        }
        code.textContent = data.final_code;
        renderTrace(data.critiques);
        setApproval(`${data.approved ? "Approved" : "Max rounds"} - ${data.round_count} round${data.round_count === 1 ? "" : "s"}`, data.approved ? "good" : "bad");
      } catch (error) {
        code.textContent = error.message;
        trace.innerHTML = '<div class="empty">Request failed. Check your provider configuration and server logs.</div>';
        setApproval("Error", "bad");
      } finally {
        run.disabled = false;
        document.querySelectorAll(".example").forEach((button) => button.disabled = false);
      }
    });
  </script>
</body>
</html>
"""
