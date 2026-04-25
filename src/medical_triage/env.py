from __future__ import annotations
import copy
from typing import Any

from .models import Patient, TriageObservation, TriageAction, TriageReward, TriageState, PatientRanking
from .grader import grade
from .patients import EASY_PATIENTS, MEDIUM_PATIENTS, HARD_PATIENTS

TASK_REGISTRY: dict[str, list[Patient]] = {
    "easy": EASY_PATIENTS,
    "medium": MEDIUM_PATIENTS,
    "hard": HARD_PATIENTS,
}

TASK_CONTEXT = {
    "easy": (
        "Community ED, suburban hospital. Moderate volume. "
        "3 patients waiting. Assign ESI 1-5 and rank by urgency."
    ),
    "medium": (
        "Urban Level-II trauma centre. Moderate surge. "
        "8 patients arrived simultaneously. Several have overlapping presentations. "
        "Careful prioritisation required."
    ),
    "hard": (
        "Level-I trauma centre. Mass-casualty drill scenario. "
        "15 patients across a spectrum of acuity — several with MISLEADING VITALS "
        "or atypical presentations. Time-critical decisions required. "
        "Some patients may look stable but have immediately life-threatening conditions."
    ),
    "custom": (
        "Custom triage scenario provided at runtime. "
        "Assign ESI 1-5 and rank by urgency."
    ),
}

class MedicalTriageEnv:
    """OpenEnv-compliant Medical Triage environment with optional custom patient sets."""

    VALID_TASKS = list(TASK_REGISTRY.keys())

    def __init__(
        self,
        task_id: str = "easy",
        custom_patients: list[Patient] | None = None,
        custom_context: str | None = None,
    ):
        if custom_patients is None and task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS} unless custom_patients is provided")

        self._custom_patients = custom_patients
        self._custom_context = custom_context or TASK_CONTEXT["custom"]
        self.task_id = "custom" if custom_patients is not None else task_id

        self._patients: list[Patient] = []
        self._step = 0
        self._done = False
        self._last_reward: TriageReward | None = None
        self._total_reward = 0.0
        self._agent_rankings: list[PatientRanking] | None = None

    def reset(self) -> TriageObservation:
        if self._custom_patients is not None:
            self._patients = copy.deepcopy(self._custom_patients)
        else:
            self._patients = copy.deepcopy(TASK_REGISTRY[self.task_id])
        self._step = 0
        self._done = False
        self._last_reward = None
        self._total_reward = 0.0
        self._agent_rankings = None
        return self._build_observation()

    def step(self, action: TriageAction) -> tuple[TriageObservation, TriageReward, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        submitted_ids = {r.patient_id for r in action.rankings}
        expected_ids = {p.patient_id for p in self._patients}
        missing = expected_ids - submitted_ids
        extra = submitted_ids - expected_ids

        info: dict[str, Any] = {
            "missing_patients": list(missing),
            "extra_patients": list(extra),
            "task_id": self.task_id,
        }
        if missing:
            info["warning"] = f"Missing rankings for: {missing}"

        reward = grade(self._patients, action.rankings)
        self._last_reward = reward
        self._total_reward += reward.total
        self._agent_rankings = action.rankings
        self._step += 1
        self._done = True

        obs = self._build_observation(action=action)
        return obs, reward, self._done, info

    def state(self) -> TriageState:
        return TriageState(
            task_id=self.task_id,
            step=self._step,
            done=self._done,
            patients=self._patients,
            agent_rankings=self._agent_rankings,
            last_reward=self._last_reward,
            total_reward=round(self._total_reward, 4),
        )

    def _build_observation(self, action: TriageAction | None = None) -> TriageObservation:
        prev = []
        if action:
            prev = [r.model_dump() for r in action.rankings]
        context = self._custom_context if self.task_id == "custom" else TASK_CONTEXT[self.task_id]
        return TriageObservation(
            task_id=self.task_id,
            step=self._step,
            patients=self._patients,
            context=context,
            action_required=(
                "Assign each patient an ESI level (1=Immediate, 5=Non-Urgent) "
                "and return them ranked from most to least urgent. "
                "Provide clinical rationale for each ranking decision."
            ),
            previous_actions=prev,
        )