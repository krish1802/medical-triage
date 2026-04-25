"""
Minimal TRL/GRPO-style training scaffold for the Medical Triage environment.
This version is intentionally hackathon-friendly: it creates prompt/reward
rollouts from the local OpenEnv environment so you can later plug the same
functions into TRL + Unsloth.

Usage:
    python train.py --task easy --episodes 20 --model gpt-4o

Notes:
- This script does NOT run full GRPO by itself.
- It prepares the exact pieces you need: prompt builder, rollout function,
  reward extraction, and training sample generation.
- Replace the placeholder section with your TRL trainer once your training
  stack is installed.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import textwrap
from typing import Any

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from openai import OpenAI
from src.medical_triage import MedicalTriageEnv, TriageAction, PatientRanking

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o"  # default, can be overridden via CLI

SYSTEM_PROMPT = textwrap.dedent("""
You are an experienced emergency department triage nurse with 15+ years of experience.
Your role is to assess patients and assign ESI (Emergency Severity Index) levels:
ESI-1: Immediate — requires immediate life-saving intervention
ESI-2: Emergent — high-risk situation; should not wait
ESI-3: Urgent — stable but needs ≥2 resources
ESI-4: Less Urgent — stable, needs 1 resource
ESI-5: Non-Urgent — stable, no resources needed

You will receive a list of patients. For each patient:
1. Assign an ESI level (1-5)
2. Provide clinical rationale referencing specific symptoms and vitals
3. Rank all patients from most urgent (ESI-1) to least urgent (ESI-5)

IMPORTANT: Prioritise life-threatening conditions even when vitals appear deceptively stable.
Watch for: aortic aneurysm (back pain + BP differential), ectopic pregnancy (missed period +
pelvic pain + haemodynamic instability), meningitis (fever + neck stiffness + photophobia),
sepsis (infection + altered mental status + tachycardia + hypotension).

Respond ONLY with valid JSON matching this schema:
{
  "rankings": [
    {
      "patient_id": "...",
      "esi_level": 1,
      "rationale": "..."
    }
  ],
  "additional_notes": "..."
}
Most urgent patient FIRST. Include ALL patients. No markdown fences.
""").strip()


def build_patient_prompt(obs) -> str:
    lines = [
        f"TRIAGE SCENARIO: {obs.context}\n",
        f"ACTION REQUIRED: {obs.action_required}\n",
        f"PATIENTS ({len(obs.patients)} total):\n",
    ]
    for p in obs.patients:
        v = p.vitals
        lines.append(f"Patient {p.patient_id}: {p.age}yo {p.sex}")
        lines.append(f" Chief complaint: {p.chief_complaint}")
        lines.append(f" Symptoms: {', '.join(p.symptoms)}")
        vitals_parts = []
        if v.heart_rate is not None:
            vitals_parts.append(f"HR {v.heart_rate}")
        if v.blood_pressure_sys is not None:
            vitals_parts.append(f"BP {v.blood_pressure_sys}/{v.blood_pressure_dia}")
        if v.respiratory_rate is not None:
            vitals_parts.append(f"RR {v.respiratory_rate}")
        if v.oxygen_saturation is not None:
            vitals_parts.append(f"SpO2 {v.oxygen_saturation}%")
        if v.temperature is not None:
            vitals_parts.append(f"Temp {v.temperature}°C")
        if v.gcs is not None:
            vitals_parts.append(f"GCS {v.gcs}")
        lines.append(f" Vitals: {', '.join(vitals_parts)}")
        lines.append(f" History: {p.history}\n")
    return "\n".join(lines)


def parse_action(raw: str, obs) -> tuple[TriageAction, dict[str, Any]]:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
        action = TriageAction(
            rankings=[PatientRanking(**r) for r in data["rankings"]],
            additional_notes=data.get("additional_notes"),
        )
        return action, data
    except Exception:
        fallback = {
            "rankings": [
                {
                    "patient_id": p.patient_id,
                    "esi_level": 3,
                    "rationale": "Fallback — parse error",
                }
                for p in obs.patients
            ],
            "additional_notes": "Fallback action due to parse failure",
        }
        action = TriageAction(
            rankings=[PatientRanking(**r) for r in fallback["rankings"]],
            additional_notes=fallback.get("additional_notes"),
        )
        return action, fallback


def rollout_once(client: OpenAI, task_id: str, model_name: str) -> dict[str, Any]:
    env = MedicalTriageEnv(task_id=task_id)
    obs = env.reset()
    prompt = build_patient_prompt(obs)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        stream=False,
    )
    raw = (completion.choices[0].message.content or "").strip()
    action, data = parse_action(raw, obs)
    _, reward, done, info = env.step(action)

    return {
        "task_id": task_id,
        "prompt": prompt,
        "response": raw,
        "parsed": data,
        "reward_total": reward.total,
        "reward_components": reward.model_dump(),
        "info": info,
        "done": done,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--out", default="train_rollouts.jsonl")
    args = parser.parse_args()

    model_name = args.model

    if not API_KEY:
        raise RuntimeError("Missing HF_TOKEN / API_KEY / OPENAI_API_KEY")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for i in range(args.episodes):
        row = rollout_once(client, args.task, model_name)
        results.append(row)
        print(f"episode={i+1} task={row['task_id']} reward={row['reward_total']:.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg = sum(r["reward_total"] for r in results) / max(len(results), 1)
    print(f"saved={args.out} avg_reward={avg:.3f}")



if __name__ == "__main__":
    main()