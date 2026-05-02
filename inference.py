import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env.environment import LicenseComplianceEnv
from env.models import Action

IMAGE_NAME = os.getenv("IMAGE_NAME") # Keeping for compatibility if needed 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "classify_licenses")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "LicenseComplianceEnv")
MAX_STEPS = 20
TEMPERATURE = 0.1
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous License Compliance agent.
    Your objective is to inspect software files and dependencies, and perform actions 
    such as classifying licenses, flagging conflicts, or generating compliance reports.
    Always reply with a strict JSON object that conforms to the Action schema.
    Allowed action_types: classify_license, flag_conflict, mark_reviewed, 
    request_clarification, add_finding, propose_remediation, generate_report.
    
    Example JSON:
    {
      "action_type": "classify_license",
      "target_id": "file_000",
      "classification": "MIT",
      "category": "permissive",
      "confidence": 0.95
    }
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure no newlines in action string
    action_str = action.replace('\n', ' ').replace('\r', '')
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    obs_json = json.dumps(obs_dict, indent=2)
    return textwrap.dedent(
        f"""
        Step: {step}
        Observation (JSON state of environment):
        {obs_json}
        
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        
        Send your specific Action as JSON. Reply EXCLUSIVELY with valid JSON.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs_dict: dict, last_reward: float, history: List[str]) -> tuple[Optional[Action], str, Optional[str]]:
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
    raw_response = ""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            # If the endpoint doesn't support json_object, you might need to remove response_format
            response_format={"type": "json_object"}
        )
        raw_response = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw_response)
        action = Action(**data)
        return action, raw_response, None
    except Exception as exc:
        print(f"[DEBUG] Model request or parsing failed: {exc}", flush=True)
        return None, raw_response or "{}", str(exc)


def main() -> None:
    api_key_to_use = API_KEY
    if not api_key_to_use:
        # Fallback dummy key to bypass initialization if using a local mock
        api_key_to_use = "dummy-key"
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key_to_use)

    # Initialize our actual environment 
    env = LicenseComplianceEnv(task_id=TASK_NAME, seed=42)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break

            # Need to pass simple dict for prompt serialization
            obs_dict = obs.model_dump(mode='json')
            
            action, raw_response, error_msg = get_model_action(client, step, obs_dict, last_reward, history)
            
            if not action:
                # If failure occurs, use a fallback action
                try:
                    action = Action(action_type="request_clarification", target_id="fallback")
                except:
                    break

            # Execute action
            try:
                obs, step_reward, done, info = env.step(action)
                float_reward = step_reward.total
            except Exception as e:
                error_msg = str(e)
                float_reward = 0.0
                done = env.done

            rewards.append(float_reward)
            steps_taken = step
            last_reward = float_reward

            action_str = json.dumps(action.model_dump(mode='json'))
            log_step(step=step, action=action_str, reward=float_reward, done=done, error=error_msg)

            history.append(f"Step {step}: {action_str} -> reward {float_reward:+.2f}")

            if done:
                break

        # Compute final validation using standard scoring methods
        final_reward = env.final_score()
        score = final_reward.total
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as general_exec:
        print(f"[DEBUG] Execution error: {general_exec}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
