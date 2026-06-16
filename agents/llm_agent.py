"""
agents/llm_agent.py — LLM-powered agent with chain-of-thought reasoning.

This is the star of your submission. Instead of hard-coded rules,
a frontier LLM reasons through the situation step by step and decides
which resource to send where — just like a real incident commander.

Key improvements over v1's inference.py:
  1. Structured observation table (not a raw Python dict dump)
  2. Chain-of-thought: model reasons before acting
  3. Explicit triage heuristics in the system prompt
  4. Handles all new zone attributes (morale, disease_risk, infra_damage)
  5. Falls back to RuleBasedAgent on any API failure
  6. Tracks reasoning history for the full episode (useful for evals)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base      import BaseAgent
from agents.rule_based import RuleBasedAgent
from env.disaster_env  import FOOD, MEDICAL, RESCUE

FALLBACK = RuleBasedAgent()

# ── System prompt ──────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI incident commander coordinating disaster relief.
Each round you allocate ONE batch of resources to ONE disaster zone.

Resources:
  0 = Food        (reduces food_need, lowers disease_risk, boosts morale)
  1 = Medical     (reduces injured count, boosts morale)
  2 = Rescue      (unblocks inaccessible zones, rescues injured people)

Quantity options:
  0 = 10 units  (conserve — use when stock is low)
  1 = 20 units  (standard)
  2 = 30 units  (maximum — use when stock is high and need is severe)

TRIAGE RULES (follow in order):
  1. If any zone has rescue_blocked=True → send Rescue there FIRST
  2. If any zone has disease_risk > 0.6 → send Food there URGENTLY
  3. Compare injured vs food_need across all zones → address the larger need
  4. Prefer zones with higher severity when tied
  5. Never send a resource to a zone that doesn't need it (waste penalty applies)
  6. Conserve resources when stock < 2 × max_quantity_option

Think step by step:
  - Identify the most critical zone and why
  - Identify the most urgent resource type and why
  - Choose the appropriate quantity
  - Output your final action as JSON

CRITICAL: Respond ONLY with a JSON object in this exact format:
{
  "reasoning": "Your step-by-step analysis (2-4 sentences)",
  "resource_type": <0|1|2>,
  "zone_id": <integer>,
  "quantity_index": <0|1|2>
}
No other text. No markdown fences. Just the JSON object."""


# ── Observation formatter ──────────────────────────────────────────────

def format_observation(env) -> str:
    """Format the env state into a clear table the LLM can reason about."""
    state = env.state()
    lines = [
        f"=== STEP {state['step']+1}/{state['max_steps']}  |  {state['difficulty'].upper()} ===",
        f"STOCK  Food:{state['food_stock']}  Medical:{state['med_stock']}  Rescue:{state['resc_stock']}",
        "",
        f"{'ID':<4} {'Zone':<8} {'Disaster':<12} {'Sev':>5} {'Injured':>8} {'Food':>6} "
        f"{'Blocked':>8} {'Disease':>8} {'Morale':>7} {'Infra':>6}",
        "-" * 72,
    ]
    for z in state["zones"]:
        blocked = "YES" if z["rescue_blocked"] else "no"
        lines.append(
            f"{z['zone_id']:<4} {z['name']:<8} {z['disaster_type']:<12} "
            f"{z['severity']:>5.2f} {z['injured']:>8} {z['food_need']:>6} "
            f"{blocked:>8} {z.get('disease_risk', 0):>8.2f} "
            f"{z.get('morale', 1):>7.2f} {z.get('infra_damage', 0):>6.2f}"
        )
    lines.append("")
    lines.append("Choose ONE action. Output JSON only.")
    return "\n".join(lines)


# ── LLM client factory ─────────────────────────────────────────────────

def _get_client():
    """Return (client, model_name) or (None, None) if not configured."""
    # Support OpenAI-compatible APIs (Scaler, OpenAI, Together, etc.)
    api_base = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key  = os.environ.get("API_KEY")      or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None, None

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=api_base if api_base else "https://api.openai.com/v1",
            api_key=api_key,
        )
        model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
        print(f"[LLMAgent] Connected — model={model}", flush=True)
        return client, model
    except ImportError:
        print("[LLMAgent] openai package not installed — using fallback", flush=True)
        return None, None


# ── Agent class ────────────────────────────────────────────────────────

class LLMAgent(BaseAgent):
    """
    Uses a frontier LLM with chain-of-thought prompting.
    Falls back to RuleBasedAgent if the API is unavailable or fails.
    """

    name        = "llm_agent"
    description = "Chain-of-thought LLM agent with triage-aware system prompt."

    def __init__(self):
        self._client, self._model = _get_client()
        self._fallback             = RuleBasedAgent()
        self.reasoning_log: list   = []   # stores (step, reasoning, action) tuples
        self._api_calls  = 0
        self._api_errors = 0

    def reset(self):
        self.reasoning_log = []
        self._api_calls    = 0
        self._api_errors   = 0

    @property
    def using_llm(self) -> bool:
        return self._client is not None

    def act(self, env) -> tuple:
        if self._client is None:
            return self._fallback.act(env)

        prompt = format_observation(env)
        self._api_calls += 1

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,    # low temp = more consistent decisions
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()

            # Strip accidental markdown fences
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()

            data    = json.loads(text)
            action  = (
                int(data["resource_type"]),
                int(data["zone_id"]),
                int(data["quantity_index"]),
            )
            reasoning = data.get("reasoning", "")

            # Validate bounds
            num_zones = len(env.zones)
            if not (0 <= action[0] <= 2 and 0 <= action[1] < num_zones and 0 <= action[2] <= 2):
                raise ValueError(f"Action out of bounds: {action}")

            self.reasoning_log.append({
                "step":      env.step_count,
                "reasoning": reasoning,
                "action":    action,
            })

            print(
                f"[LLM step={env.step_count}] "
                f"res={action[0]} zone={action[1]} qty={action[2]} | "
                f"{reasoning[:80]}...",
                flush=True,
            )
            return action

        except Exception as e:
            self._api_errors += 1
            print(f"[LLMAgent] API error (fallback): {e}", flush=True)
            return self._fallback.act(env)

    def stats(self) -> dict:
        return {
            "api_calls":  self._api_calls,
            "api_errors": self._api_errors,
            "error_rate": round(self._api_errors / max(1, self._api_calls), 3),
        }
