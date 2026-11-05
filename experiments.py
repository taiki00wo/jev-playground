"""飲食店の売上低下の原因を、Jev に段階的に聞いていく実験。

プレイグラウンドで手作業で試していた流れをスクリプトにしたもの。
  1. 答えがはっきりしたデータ（Choice / Score / Noul）
  2. 原因が混ざったデータ（比較の基準なし → あり）
  3. 原因ごとに Noul で聞く（質問に観察を混ぜる）
  4. 変化率を計算して渡し、判断だけを聞く

使い方: uv run experiments.py
"""

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv(Path(__file__).parent / ".env")

CAUSES = {
    "service": "待ち時間や接客の悪化で、常連客が離れている",
    "competitor": "近くの競合店に、新規客が流れている",
    "weather": "雨の日が多く、来店が減っている",
    "menu": "メニュー変更で、客単価が下がっている",
}
MAIN_ISSUE = Choice(instructions="この店の売上が下がった主な原因はどれか", criteria=CAUSES)
SEVERITY = Score(
    instructions="この店の売上改善の緊急度",
    criteria=["問題なし", "要観察", "すぐに対策が必要"],
)


def store(**kw):
    """売上 = 客数 × 客単価 になるように売上を計算して足す。"""
    kw["sales"] = kw["customers"] * kw["avg_spend"]
    if "customers_prev_month" in kw and "avg_spend_prev_month" in kw:
        kw["sales_prev_month"] = kw["customers_prev_month"] * kw["avg_spend_prev_month"]
    return kw


def pct(now, prev):
    return round((now / prev - 1) * 100)


# 1. 待ち時間が延び、リピート率が落ちた店（原因は service がはっきり）
STEP1 = store(
    store_name="駅前店",
    period="今月",
    customers=2100,
    customers_prev_month=2600,
    avg_spend=1500,
    avg_spend_prev_month=1500,
    repeat_rate=31,
    repeat_rate_prev_month=46,
    avg_wait_min=24,
    avg_wait_min_prev_month=9,
    review_rating=3.6,
    review_rating_prev_month=4.3,
    rainy_days=5,
)

# 2. 雨の日の増加とメニュー変更が重なった店。雨の日数だけ前月の値がない
STEP2_NO_BASELINE = store(
    store_name="商店街店",
    period="今月",
    customers=2290,
    customers_prev_month=2600,
    avg_spend=1380,
    avg_spend_prev_month=1500,
    repeat_rate=45,
    repeat_rate_prev_month=46,
    avg_wait_min=9,
    avg_wait_min_prev_month=9,
    review_rating=4.2,
    review_rating_prev_month=4.3,
    rainy_days=13,
)
STEP2_WITH_BASELINE = {**STEP2_NO_BASELINE, "rainy_days_prev_month": 4}

# 3. 原因ごとの Noul。質問文にデータの観察を混ぜた版
NOULS_WITH_OBSERVATION = {
    "cause_service": Noul(instructions="`avg_wait_min`が延び`repeat_rate`が下がっており、常連客が離れている"),
    "cause_competitor": Noul(instructions="`customers`が`customers_prev_month`より減っており、競合店に客が流れている"),
    "cause_weather": Noul(instructions="`rainy_days`が多く、雨で来店が減っている"),
    "cause_menu": Noul(instructions="`avg_spend`が`avg_spend_prev_month`より下がっており、メニュー変更で客単価が落ちている"),
}

# 4. 変化率を計算して渡し、質問からは観察を外して判断だけを聞く
s = STEP2_WITH_BASELINE
STEP4 = {
    **s,
    "sales_change_pct": pct(s["sales"], s["sales_prev_month"]),
    "customers_change_pct": pct(s["customers"], s["customers_prev_month"]),
    "avg_spend_change_pct": pct(s["avg_spend"], s["avg_spend_prev_month"]),
    "repeat_rate_change_pt": s["repeat_rate"] - s["repeat_rate_prev_month"],
    "rainy_days_change": s["rainy_days"] - s["rainy_days_prev_month"],
}
NOULS_JUDGMENT_ONLY = {
    "cause_service": Noul(instructions="売上低下の大きな要因は、待ち時間や接客の悪化で常連客が離れたことである"),
    "cause_competitor": Noul(instructions="売上低下の大きな要因は、競合店に新規客が流れたことである"),
    "cause_weather": Noul(instructions="売上低下の大きな要因は、雨の日が増えて来店が減ったことである"),
    "cause_menu": Noul(instructions="売上低下の大きな要因は、メニュー変更で客単価が下がったことである"),
}

STEPS = [
    ("1. 答えがはっきりしたデータ", STEP1, {"main_issue": MAIN_ISSUE, "severity": SEVERITY, "long_wait": Noul(instructions="`avg_wait_min`が長すぎて、客が不満を持っている")}),
    ("2a. 原因が混ざったデータ（雨の日数の基準なし）", STEP2_NO_BASELINE, {"main_issue": MAIN_ISSUE}),
    ("2b. 原因が混ざったデータ（雨の日数の基準あり）", STEP2_WITH_BASELINE, {"main_issue": MAIN_ISSUE}),
    ("3. 原因ごとの Noul（観察を混ぜた質問）", STEP2_WITH_BASELINE, {"main_issue": MAIN_ISSUE, **NOULS_WITH_OBSERVATION}),
    ("4. 変化率を渡し、判断だけを聞く", STEP4, NOULS_JUDGMENT_ONLY),
]


def show(res):
    out = {}
    for k, a in res.choices.items():
        out[k] = {"choice": a.choice, "confidence": a.confidence, "probabilities": a.probabilities}
    for k, a in res.scores.items():
        out[k] = {"score": a.score, "confidence": a.confidence, "probabilities": a.probabilities}
    for k, a in res.nouls.items():
        out[k] = {"noul": a.noul}
    return out


def main():
    log = []
    with TypeSafeClient(model="jev-latest") as api:
        for title, state, questions in STEPS:
            res = api.system_one(state=state, questions=questions)
            answers = show(res)
            print(f"\n=== {title} ===")
            print(json.dumps(answers, ensure_ascii=False, indent=2))
            print(f"input_tokens={res.usage.input_tokens}")
            log.append({"step": title, "state": state, "answers": answers, "input_tokens": res.usage.input_tokens})

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"experiments-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n詳細: {path}")


if __name__ == "__main__":
    main()
