import os
import json
import argparse
import time
import re
import random
import openai

from TextEnv_v2 import LightBulbEnv

# ------------------- 설정 -------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4.1-mini")
parser.add_argument("--api_key", type=str, default=None,
                    help="OpenAI API key (기본값: OPENAI_API_KEY 환경변수)")
parser.add_argument("--num_test_data", type=int, default=111)
parser.add_argument("--save_file", type=str, default="output/openai_gpt41mini.json")
parser.add_argument("--max_steps", type=int, default=200)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# API 키 초기화
api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key가 필요합니다. --api_key 옵션 또는 OPENAI_API_KEY 환경변수를 설정하세요.")

client = openai.OpenAI(api_key=api_key)
print(f"Using OpenAI API with model: {args.model}")

# ------------------- 유틸 함수 -------------------

def extract_action(text: str) -> str:
    """<action> 태그에서 행동을 추출한다."""
    m = re.search(r"<action>(.*?)</action>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def generate_prompt(env, history, feedback, goal_state_str=None):
    """LLM 입력 prompt 생성"""
    grid_text = env.return_obs()
    history_text = "\n".join(history)

    if goal_state_str is not None:
        goal_desc = (
            f"Your mission is to reach the target bulb state: \"{goal_state_str}\" (💡=ON, ○=OFF).\n"
            "However, the accessibility of the bulbs is based on the current condition of other bulbs.\n"
            "You need to learn the hidden rule behind the environment and complete the task."
        )
    else:
        goal_desc = (
            "Your mission is to light on all the bulbs.\n"
            "However, the accessibility of the bulbs is based on the current condition of other bulbs.\n"
            "You need to learn the hidden rule behind the environment and complete the task."
        )

    prompt = f"""
You are an intelligent agent.

### Goal:
{goal_desc}

### Action Space:
The action space is based on the index of bulbs. For example, you would like to light on / off the first bulb, you should \
output <action>0</action> to toggle the state of the bulb. 

### History Action and Feedback:
{history_text}

### Current State:
{grid_text}

Now think step by step and choose the next action to act in the environment.
You are encouraged to act actively to derive the environment dynamics.
Output ONLY one action in the format: <action>n</action>
"""
    return prompt.strip()


def call_openai(prompt: str, max_retries: int = 8, base_delay: float = 1.0) -> tuple[str, int]:
    """
    OpenAI API를 호출하고 응답 텍스트와 토큰 수를 반환한다.
    Rate limit(429) 에러 시 exponential backoff으로 재시도한다.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            text = response.choices[0].message.content.strip()
            token_num = response.usage.completion_tokens
            return text, token_num

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"  [rate_limit] 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
            time.sleep(delay)

        except openai.APIStatusError as exc:
            if exc.status_code == 429 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  [rate_limit] 429 – retry {attempt + 1}/{max_retries} in {delay:.1f}s …")
                time.sleep(delay)
            else:
                raise


# ------------------- 메인 로직 -------------------

def infer():
    with open("../test_data/turnonlights/test_turnonlights_lite_251030_augmented.json", "r") as f:
        test_data = json.load(f)
    args.num_test_data = len(test_data)

    results = []

    for env_idx in range(args.num_test_data):
        print(f"\n===== [Env {env_idx + 1}/{args.num_test_data}] =====")
        d = test_data[env_idx]
        goal_state = d.get("goal_state", None)
        env = LightBulbEnv(custom_logic=d["custom_logic"], num_bulbs=d["level"], goal_state=goal_state)
        goal_state_str = " ".join("💡" if b else "○" for b in goal_state) if goal_state is not None else None

        history = []
        feedback = ""
        traj = {
            "env_id": env_idx,
            "level": d["level"],
            "custom_logic": d["custom_logic"],
            "goal_state": goal_state,
            "initial_state": env.return_obs(),
            "num_steps": 0,
            "steps": [],
            "token_num_total": 0,
            "success": False,
        }
        done = False
        token_num_total = 0

        for step in range(args.max_steps):
            user_prompt = generate_prompt(env, history, feedback, goal_state_str=goal_state_str)

            action_text, token_num_step = call_openai(user_prompt)
            token_num_total += token_num_step

            print("-" * 20)
            action_str = extract_action(action_text + "</action>")

            # 행동 파싱
            try:
                action = int(action_str)
                assert action in list(range(env.num_bulbs))
            except Exception:
                print(f"[WARN] Invalid action output: {action_text}")
                traj["steps"].append(
                    {
                        "step": step,
                        "raw_output": action_text,
                        "token_num": token_num_step,
                        "action": None,
                        "error": "invalid_action",
                    }
                )
                continue

            # 환경과 상호작용
            obs, feedback, done, _ = env.step(action)
            env_state = obs
            history.append(f"Action: {action}, Feedback: {feedback}, State: {obs}")

            traj["steps"].append(
                {
                    "step": step,
                    "action": action,
                    "raw_output": action_text,
                    "token_num": token_num_step,
                    "grid": env_state,
                    "feedback": feedback,
                }
            )

            print(f"Step {step}: Action={action}")
            print(feedback)
            print(env_state)

            if done:
                print("Mission complete!")
                traj["success"] = True
                traj["num_steps"] = step
                break

        traj["token_num_total"] = token_num_total
        results.append(traj)

        # 중간 저장
        os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        with open(args.save_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nDone! Results saved to {args.save_file}")


if __name__ == "__main__":
    infer()
