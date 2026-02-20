"""
test_turnonlights_lite_251030.json 데이터셋에
각 custom_logic에서 BFS로 도달 가능한 고유 obs 리스트와
자동으로 선택된 goal_state를 추가하는 스크립트.

goal_state 선택 기준:
  - 초기 상태(전부 False)로부터 BFS 거리가 max_dist * goal_dist_ratio 이상인 상태 중
    랜덤(seed 고정)으로 하나 선택.
  - 해당하는 상태가 없으면 가장 먼 거리의 상태를 선택.

출력: test_data/turnonlights/test_turnonlights_lite_251030_augmented.json
"""

import json
import random
from collections import deque


def bfs_reachable(custom_logic: dict, num_bulbs: int):
    """
    BFS로 도달 가능한 모든 (obs, distance) 쌍을 반환.
    obs: list[bool] (_get_obs()와 동일한 순서)
    distance: 초기 상태(전부 False)로부터의 최단 step 수
    """
    bulb_names = [f"B{i}" for i in range(num_bulbs)]
    init_state = {b: False for b in bulb_names}
    init_key = tuple(init_state[b] for b in bulb_names)

    visited = {init_key: 0}  # key -> distance
    queue = deque([(init_state, 0)])
    results = []  # (obs_list, distance)

    while queue:
        state, dist = queue.popleft()
        state_key = tuple(state[b] for b in bulb_names)
        results.append((list(state_key), dist))

        for bulb in bulb_names:
            if bulb not in custom_logic:
                continue
            expr = custom_logic[bulb]
            try:
                can_toggle = bool(eval(expr, {"__builtins__": {}}, state.copy()))
            except Exception:
                can_toggle = False

            if can_toggle:
                new_state = state.copy()
                new_state[bulb] = not new_state[bulb]
                new_key = tuple(new_state[b] for b in bulb_names)

                if new_key not in visited:
                    visited[new_key] = dist + 1
                    queue.append((new_state, dist + 1))

    return results


def select_goal_state(
    reachable: list,
    num_bulbs: int,
    rng: random.Random,
    goal_dist_ratio: float = 0.5,
    min_dist: int = 3,
):
    """
    reachable: [(obs_list, distance), ...]
    goal_dist_ratio: max_dist * ratio 이상 거리 상태에서 goal 선택
    min_dist: 최소 거리 하한 (ratio 기준보다 작으면 이 값 사용)

    초기 상태(all False) 및 all True 상태는 제외.
    """
    if len(reachable) <= 1:
        return reachable[0][0]

    max_dist = max(d for _, d in reachable)
    threshold = max(min_dist, int(max_dist * goal_dist_ratio))

    all_false = [False] * num_bulbs
    all_true = [True] * num_bulbs

    candidates = [
        obs for obs, dist in reachable
        if dist >= threshold and obs != all_false and obs != all_true
    ]

    if not candidates:
        # threshold를 만족하는 후보가 없으면 가장 먼 상태 중 하나 선택
        candidates = [
            obs for obs, dist in reachable
            if dist == max_dist and obs != all_false
        ]

    if not candidates:
        # 마지막 fallback
        candidates = [obs for obs, dist in reachable if obs != all_false]

    return rng.choice(candidates)


def main(
    input_path: str = "test_data/turnonlights/test_turnonlights_lite_251030.json",
    output_path: str = "test_data/turnonlights/test_turnonlights_lite_251030_augmented.json",
    goal_dist_ratio: float = 0.5,
    min_dist: int = 3,
    seed: int = 42,
):
    with open(input_path, "r") as f:
        dataset = json.load(f)

    augmented = []
    rng = random.Random(seed)

    for entry in dataset:
        num_bulbs = entry["level"]
        custom_logic = entry["custom_logic"]

        reachable = bfs_reachable(custom_logic, num_bulbs)

        unique_obs = [obs for obs, _ in reachable]
        goal_state = select_goal_state(
            reachable, num_bulbs, rng,
            goal_dist_ratio=goal_dist_ratio,
            min_dist=min_dist,
        )

        max_dist = max(d for _, d in reachable)
        goal_dist = next(d for obs, d in reachable if obs == goal_state)

        augmented.append({
            **entry,
            "goal_state": goal_state,
            "reachable_obs": unique_obs,
        })

        print(
            f"idx={entry.get('idx', '?'):>3} | level={num_bulbs} "
            f"| reachable={len(unique_obs):>4} | max_dist={max_dist} "
            f"| goal_dist={goal_dist} | goal={goal_state}"
        )

    with open(output_path, "w") as f:
        json.dump(augmented, f, indent=4)

    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
