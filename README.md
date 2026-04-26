# Medical Triage

A hackathon-friendly medical triage environment for evaluating and training LLMs on emergency-department style patient prioritization. The project simulates triage scenarios, asks a model to assign ESI levels, ranks patients by urgency, and scores the response with a structured reward pipeline.

## What this project does

This repository provides a lightweight environment for **medical triage reasoning**:

- Generates prebuilt triage scenarios (`easy`, `medium`, `hard`).
- Presents multiple patients with symptoms, vitals, and history.
- Prompts an LLM to return structured JSON with:
  - ESI level per patient
  - Clinical rationale
  - Final urgency ranking
- Grades the output and stores rollout data for later RL / GRPO-style training.

The codebase is designed to be simple enough for hackathons while still being useful as a foundation for experimentation with triage agents.

## Repository structure

```text
medical-triage/
├── logs/
│   └── triage_log.jsonl
├── src/medical_triage/
│   ├── __init__.py
│   ├── env.py
│   ├── grader.py
│   ├── models.py
│   ├── patients.py
│   └── runtime_config.py
├── .github/workflows/
│   └── train.yml
├── requirements.txt
├── train.py
└── train_rollouts.jsonl
```

## Core components

### `train.py`
Main training / rollout scaffold.

- Builds the triage prompt from patient observations.
- Sends the prompt to a chat model.
- Parses the model response into structured triage actions.
- Runs the environment step.
- Writes rollouts to `train_rollouts.jsonl`.

### `src/medical_triage/env.py`
Defines the `MedicalTriageEnv` environment.

- Supports `easy`, `medium`, and `hard` tasks.
- Resets scenarios and returns triage observations.
- Accepts a ranked triage action.
- Returns reward and grading metadata.

### `src/medical_triage/grader.py`
Evaluates how well the model prioritized patients and assigned ESI levels.

### `src/medical_triage/patients.py`
Contains the predefined patient cases for each difficulty level.

### `logs/triage_log.jsonl`
Stores triage logs in JSONL format.

## Hugging Face logging note

If you are running the app version on Hugging Face Spaces, the **logs are created there from `app.py`**, not only in the local training setup.

Relevant Space:

- https://huggingface.co/spaces/purunjaybhardwaj/medical-triage-hackathon/tree/main

That means when the hosted app is used, log output is generated inside the Hugging Face Space environment. If you want to inspect live or hosted logs, check the Space files/runtime instead of expecting all logs to appear only in this GitHub repository.

## Installation

```bash
pip install -r requirements.txt
```

## Environment variables

The project expects an API key through one of these environment variables:

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `API_KEY`

## Run a rollout

Example:

```bash
python train.py --task easy --episodes 20 --model gpt-4o
```

### Arguments

- `--task`: `easy`, `medium`, or `hard`
- `--episodes`: number of rollout episodes to generate
- `--model`: model name to call
- `--out`: output file for rollout logs, defaults to `train_rollouts.jsonl`

## Expected model output

The agent is instructed to return JSON in this shape:

```json
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
```

The patients should be ordered from **most urgent first** to **least urgent last**.

## Use cases

- LLM evaluation on clinical prioritization tasks
- RL / GRPO-style rollout generation
- Prompt engineering for medical reasoning
- Hackathon demos for emergency triage workflows
- Benchmarking structured decision-making agents
