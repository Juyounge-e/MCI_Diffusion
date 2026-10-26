"""구 코어 vs 고속 코어 동치검증 (MCI_ADV 판) — 한 프로세스에서 양쪽을 굴려 비교.

MCI_UAV 의 `verify/head_to_head_heur.py` 는 rl_src 드라이버(`viper_distill.make_feature_env`)
경로를 검증한다. 이 저장소에는 rl_src 가 없고 진입점이 `src/sim_src/main.py` 하나이므로,
그 실행 루프와 **같은 순서로** 시나리오·규칙·환경을 만들어 지표 배열을 비교한다.

핵심: 구 코어는 flat import(`import ScenarioManager`), 고속 코어는 패키지 상대 import
(`sim_src_upgrade.core.ScenarioManager`) → 한 프로세스에 동시 로드가 가능하다.

비교 지표는 `main.py.run()` 이 파일로 쓰는 다섯 개와 같다:
    reward(합), time, PDR, reward_woG, PDR_woG   — 규칙 × 에피소드 전부.

사용
----
    python src/sim_src_upgrade/verify/sim_equivalence.py \
        --config scenarios/exp_.../\\(37.5,127.0\\)/config_....yaml --n_eps 3
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import random
import sys

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))  # → src/

from sim_src_upgrade._paths import ensure_paths  # noqa: E402

ensure_paths(rl=False, old_sim=True)

METRIC_NAMES = ("reward", "time", "pdr", "reward_woG", "pdr_woG")


def _load_core(kind: str):
    """('old'|'fast') 코어의 (ScenarioManager, RuleManager, MCIEnvironment_gym) 반환."""
    if kind == "old":
        sm = importlib.import_module("ScenarioManager")
        rm = importlib.import_module("RuleManager")
        env = importlib.import_module("MCIEnvironment_gymnasium")
    elif kind == "fast":
        sm = importlib.import_module("sim_src_upgrade.core.ScenarioManager")
        rm = importlib.import_module("sim_src_upgrade.core.RuleManager")
        env = importlib.import_module("sim_src_upgrade.core.MCIEnvironment_gymnasium")
    else:
        raise ValueError(f"알 수 없는 코어: {kind}")
    return sm.ScenarioManager, rm.RuleManager, env.MCIEnvironment_gym


def run_core(config_path: str, kind: str, n_eps: int, n_rules: int | None = None) -> np.ndarray:
    """한 코어로 (규칙수, n_eps, 5) 지표 배열 계산. main.py.run() 과 동일한 시드·순서."""
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    ScenarioManagerC, RuleManagerC, EnvC = _load_core(kind)

    cfg_run = configs["run_setting"]
    init_seed = cfg_run["random_seed"]
    if init_seed is not None:
        random.seed(init_seed)
        np.random.seed(init_seed)
        rng = np.random.default_rng(init_seed)
    else:
        rng = np.random.default_rng()

    # main.py 와 동일 순서: ScenarioManager → RuleManager → Env
    with contextlib.redirect_stdout(io.StringIO()):
        s_manager = ScenarioManagerC(configs, rng=rng)
        scenario = s_manager.scenario
        r_manager = RuleManagerC(configs["rule_info"], scenario=scenario, rng=rng)
        rules = r_manager.rules
        env = EnvC(scenario=scenario, rng=rng,
                   rule_test=cfg_run["rule_test"], eval_mode=cfg_run["eval_mode"])

    if n_rules is not None:
        rules = rules[:n_rules]

    def safe_pdr(saved, preventable):
        return 0.0 if preventable <= 0 else 1 - saved / preventable

    out = np.zeros((len(rules), n_eps, len(METRIC_NAMES)), dtype=np.float64)
    for it in range(1, n_eps + 1):
        for r_idx, rule in enumerate(rules):
            # main.py.set_random_seed() 와 동일 (seed = iter + init_seed)
            seed = it + init_seed
            random.seed(seed)
            np.random.seed(seed)
            rng_i = np.random.default_rng(seed)
            s_manager.set_seed(rng_i)
            for r in rules:
                r.set_seed(rng_i)
            env.set_seed(rng_i)

            with contextlib.redirect_stdout(io.StringIO()):
                obs, _ = env.reset()
                done = False
                cumul, cumul_woG = 0.0, 0.0
                info = {"time": 0.0}
                while not done:
                    action = rule.select(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    done = done or truncated
                    cumul += reward
                    cumul_woG += info.get("r_woG", 0.0)

            out[r_idx, it - 1] = (
                cumul,
                info["time"],
                safe_pdr(cumul, env.preventable),
                cumul_woG,
                safe_pdr(cumul_woG, env.preventable_woG),
            )
    return out


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """두 지표 배열 비교 결과 dict (판정 + 최대 절대차)."""
    if a.shape != b.shape:
        return {"pass": False, "reason": f"shape 불일치 {a.shape} vs {b.shape}", "max_abs_diff": None}
    diff = np.abs(a - b)
    max_d = float(diff.max()) if diff.size else 0.0
    per_metric = {METRIC_NAMES[i]: float(diff[..., i].max()) for i in range(a.shape[-1])}
    return {
        "pass": bool(np.array_equal(a, b)),
        "n_elements": int(a.size),
        "max_abs_diff": max_d,
        "max_abs_diff_per_metric": per_metric,
        "n_mismatch": int((diff > 0).sum()),
    }


def preflight(config_path: str, n_eps: int = 3, n_rules: int = 2, verbose: bool = True) -> dict:
    """작업 전 사전점검 — 소규모(규칙 n_rules × n_eps)로 구·신 코어 지표 완전일치 확인."""
    old = run_core(config_path, "old", n_eps=n_eps, n_rules=n_rules)
    fast = run_core(config_path, "fast", n_eps=n_eps, n_rules=n_rules)
    res = compare(old, fast)
    if verbose:
        tag = "PASS" if res["pass"] else "FAIL"
        print(f"[preflight] {tag} — 규칙 {n_rules} × 에피 {n_eps}, "
              f"원소 {res['n_elements']}개, 최대차이 {res['max_abs_diff']}")
        if not res["pass"]:
            print(f"[preflight] 불일치 원소 {res['n_mismatch']}개, "
                  f"지표별 최대차이 {res['max_abs_diff_per_metric']}")
    return res


def main() -> int:
    p = argparse.ArgumentParser(description="구 코어 vs 고속 코어 시뮬 동치검증 (MCI_ADV)")
    p.add_argument("--config", required=True, help="시나리오 config_*.yaml 경로")
    p.add_argument("--n_eps", type=int, default=3, help="에피소드 수 (기본 3)")
    p.add_argument("--n_rules", type=int, default=None, help="규칙 수 제한 (기본: 전부, 보통 64)")
    p.add_argument("--json_out", default=None, help="비교결과 JSON 저장 경로")
    args = p.parse_args()

    print(f"[G0] 원본 드리프트 검사...")
    from sim_src_upgrade import origin_sync
    origin_sync.check()
    print("[G0] PASS")

    import time
    t0 = time.time(); old = run_core(args.config, "old", args.n_eps, args.n_rules); t_old = time.time() - t0
    t0 = time.time(); fast = run_core(args.config, "fast", args.n_eps, args.n_rules); t_fast = time.time() - t0

    res = compare(old, fast)
    res.update({"wall_old_s": round(t_old, 2), "wall_fast_s": round(t_fast, 2),
                "speedup": round(t_old / t_fast, 3) if t_fast > 0 else None,
                "shape": list(old.shape), "config": args.config})
    print(f"\n구 코어 : {t_old:.2f}s")
    print(f"고속코어: {t_fast:.2f}s  ({res['speedup']}×)")
    print(f"지표 배열 {old.shape} = {res['n_elements']}개 원소")
    print(f"판정: {'PASS (비트동일)' if res['pass'] else 'FAIL'} — 최대 절대차 {res['max_abs_diff']}")
    if not res["pass"]:
        print(f"  지표별 최대차이: {res['max_abs_diff_per_metric']}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"결과 저장: {args.json_out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
