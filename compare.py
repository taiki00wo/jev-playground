"""Jev と LLM（Ollama のローカルモデル）で、飲食店の売上低下の主な原因の判定を比べる。

正解ラベル付きのダミー店舗データを作り、同じ問いを両方に投げて
正解率・速度・答えのぶれ・確信度の当てになり具合・費用を比べる。

使い方:
  uv run compare.py                      # 両方を実行（.env に TYPESAFE_API_KEY が必要）
  uv run compare.py --only llm           # LLM だけ（無料・API キー不要）
  uv run compare.py --limit 4 --repeats 1  # 少しだけ試す
"""

import argparse
import json
import random
import statistics
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")  # TYPESAFE_API_KEY を .env から読む

CAUSES = {
    "service": "待ち時間や接客の悪化で、常連客が離れている",
    "competitor": "近くの競合店に、新規客が流れている",
    "weather": "雨の日が多く、来店が減っている",
    "menu": "メニュー変更で、客単価が下がっている",
}
INSTRUCTIONS = "この店の売上が前月より下がった主な原因はどれか"
JEV_USD_PER_MTOK = 0.042  # 入力100万トークンあたり。出力は無料（docs.typesafe.ai/models）
HIGH_CONFIDENCE = 0.7


# ---------- ダミーデータ ----------

def pct(now, prev):
    return round((now / prev - 1) * 100)


def make_store(rng, cause):
    """原因ごとに、その原因らしい数値の動きを持つ店を1件作る。"""
    customers_prev = rng.randint(1800, 3200)
    repeat_rate_prev = rng.uniform(0.40, 0.55)
    avg_spend_prev = rng.randrange(1200, 2500, 10)
    wait_prev = rng.randint(6, 12)
    review_prev = round(rng.uniform(4.0, 4.5), 1)
    rainy_prev = rng.randint(3, 7)

    # 原因に関係ない指標は ±5% 程度の揺れにとどめる
    repeat_r, new_r, spend_r = (rng.uniform(0.95, 1.05) for _ in range(3))
    wait, review, rainy = wait_prev + rng.randint(-1, 1), review_prev, rainy_prev + rng.randint(-2, 2)
    menu_changed = rng.random() < 0.3  # メニュー変更はほかの原因の店でもときどき起きる

    if cause == "service":
        repeat_r = rng.uniform(0.6, 0.8)
        wait = round(wait_prev * rng.uniform(1.8, 2.8))
        review = round(review_prev - rng.uniform(0.4, 0.8), 1)
    elif cause == "competitor":
        new_r = rng.uniform(0.5, 0.7)
    elif cause == "weather":
        repeat_r = new_r = rng.uniform(0.82, 0.92)
        rainy = rainy_prev + rng.randint(6, 10)
    elif cause == "menu":
        spend_r = rng.uniform(0.75, 0.88)
        menu_changed = True

    repeat_prev = round(customers_prev * repeat_rate_prev)
    new_prev = customers_prev - repeat_prev
    repeat_now, new_now = round(repeat_prev * repeat_r), round(new_prev * new_r)
    customers = repeat_now + new_now
    avg_spend = round(avg_spend_prev * spend_r, -1)

    state = {
        "store_name": "",
        "period": "今月",
        "sales": customers * avg_spend,
        "sales_prev_month": customers_prev * avg_spend_prev,
        "sales_change_pct": pct(customers * avg_spend, customers_prev * avg_spend_prev),
        "customers": customers,
        "customers_prev_month": customers_prev,
        "customers_change_pct": pct(customers, customers_prev),
        "new_customers": new_now,
        "new_customers_prev_month": new_prev,
        "new_customers_change_pct": pct(new_now, new_prev),
        "repeat_rate": round(repeat_now / customers * 100),
        "repeat_rate_prev_month": round(repeat_rate_prev * 100),
        "avg_spend": avg_spend,
        "avg_spend_prev_month": avg_spend_prev,
        "avg_spend_change_pct": pct(avg_spend, avg_spend_prev),
        "avg_wait_min": wait,
        "avg_wait_min_prev_month": wait_prev,
        "review_rating": review,
        "review_rating_prev_month": review_prev,
        "rainy_days": rainy,
        "rainy_days_prev_month": rainy_prev,
        "menu_changed_this_month": menu_changed,
    }
    return {"id": "", "truth": cause, "state": state}


def make_dataset(n_per_cause, seed):
    rng = random.Random(seed)
    stores = [make_store(rng, c) for c in CAUSES for _ in range(n_per_cause)]
    rng.shuffle(stores)
    for i, s in enumerate(stores, 1):
        s["id"] = s["state"]["store_name"] = f"店舗{i:02d}"
    return stores


# ---------- 判定役 ----------

def run_jev(client, store):
    from typesafe_sdk import Choice

    start = time.perf_counter()
    res = client.system_one(
        state=store["state"],
        questions={"main_issue": Choice(instructions=INSTRUCTIONS, criteria=CAUSES)},
    )
    latency = time.perf_counter() - start
    answer = res.choices["main_issue"]
    return {
        "choice": answer.choice,
        "confidence": answer.confidence,
        "probabilities": answer.probabilities,
        "latency_s": latency,
        "input_tokens": res.usage.input_tokens,
    }


LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": list(CAUSES)},
        "confidence": {"type": "number"},
    },
    "required": ["choice", "confidence"],
}


def run_llm(model, store):
    options = "\n".join(f"- {k}: {v}" for k, v in CAUSES.items())
    prompt = (
        f"以下は飲食店の月次データです。{INSTRUCTIONS}。\n"
        "次の選択肢から1つ選び、その判断にどれくらい自信があるかを0〜1の数値で答えてください。\n\n"
        f"選択肢:\n{options}\n\n"
        f"データ:\n{json.dumps(store['state'], ensure_ascii=False, indent=2)}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": LLM_SCHEMA,  # 出力を選択肢に縛る（Ollama の構造化出力）
        "stream": False,
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        res = json.load(r)
    latency = time.perf_counter() - start

    try:
        out = json.loads(res["message"]["content"])
        choice = out["choice"] if out["choice"] in CAUSES else None
        confidence = float(out["confidence"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        choice, confidence = None, None
    return {
        "choice": choice,
        "confidence": confidence,
        "raw": res["message"]["content"],
        "latency_s": latency,
        "input_tokens": res.get("prompt_eval_count"),
    }


# ---------- 集計 ----------

def summarize(name, stores, runs):
    """runs[store_id] = その店に対する繰り返し実行の結果リスト"""
    all_runs = [(s, r) for s in stores for r in runs[s["id"]]]
    valid = [(s, r) for s, r in all_runs if r["choice"] is not None]
    correct = [r["choice"] == s["truth"] for s, r in valid]
    latencies = [r["latency_s"] for _, r in all_runs]
    consistent = sum(len({r["choice"] for r in runs[s["id"]]}) == 1 for s in stores)

    def acc(rows):
        return f"{sum(r['choice'] == s['truth'] for s, r in rows) / len(rows):.0%}（{len(rows)}件）" if rows else "-"

    high = [(s, r) for s, r in valid if r["confidence"] is not None and r["confidence"] >= HIGH_CONFIDENCE]
    low = [(s, r) for s, r in valid if r["confidence"] is not None and r["confidence"] < HIGH_CONFIDENCE]
    tokens = sum(r["input_tokens"] or 0 for _, r in all_runs)

    return {
        "判定役": name,
        "正解率": f"{sum(correct) / len(all_runs):.0%}",
        "無効な出力": f"{len(all_runs) - len(valid)}件",
        "答えが毎回同じ": f"{consistent}/{len(stores)}件",
        f"確信度≥{HIGH_CONFIDENCE}の正解率": acc(high),
        f"確信度<{HIGH_CONFIDENCE}の正解率": acc(low),
        "応答時間（中央値）": f"{statistics.median(latencies) * 1000:.0f}ms",
        "入力トークン合計": tokens,
        "費用": f"${tokens / 1e6 * JEV_USD_PER_MTOK:.5f}" if name == "Jev" else "$0（ローカル）",
    }


def print_table(rows):
    keys = list(rows[0])
    width = max(len(k) for k in keys) + 2
    for k in keys:
        print(f"{k:<{width}}" + "".join(f"{str(r[k]):<26}" for r in rows))


def print_per_store(stores, results):
    names = list(results)
    print("\n店舗   正解         " + "".join(f"{n:<28}" for n in names))
    for s in stores:
        line = f"{s['id']} {s['truth']:<12} "
        for n in names:
            choices = Counter(r["choice"] for r in results[n][s["id"]])
            top, _ = choices.most_common(1)[0]
            mark = "○" if top == s["truth"] else "×"
            spread = "" if len(choices) == 1 else f" ぶれ{dict(choices)}"
            line += f"{mark} {str(top)}{spread}".ljust(28)
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["jev", "llm"], help="片方だけ実行する")
    p.add_argument("--llm-model", default="llama3:8b")
    p.add_argument("--per-cause", type=int, default=5, help="原因ごとの店舗数（合計はこの4倍）")
    p.add_argument("--repeats", type=int, default=3, help="同じ店を何回判定させるか（ぶれを見る）")
    p.add_argument("--limit", type=int, help="先頭から何件だけ使うか")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stores = make_dataset(args.per_cause, args.seed)[: args.limit]
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "stores.json").write_text(json.dumps(stores, ensure_ascii=False, indent=2))

    runners = {}
    if args.only != "llm":
        from typesafe_sdk import TypeSafeClient

        client = TypeSafeClient(model="jev-latest")
        runners["Jev"] = lambda s: run_jev(client, s)
    if args.only != "jev":
        runners[f"LLM({args.llm_model})"] = lambda s: run_llm(args.llm_model, s)

    results = {}
    for name, run in runners.items():
        results[name] = {}
        for i, s in enumerate(stores, 1):
            results[name][s["id"]] = [run(s) for _ in range(args.repeats)]
            print(f"\r{name}: {i}/{len(stores)}", end="", flush=True)
        print()

    print()
    print_table([summarize(n, stores, results[n]) for n in results])
    print_per_store(stores, results)

    path = out_dir / f"results-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"args": vars(args), "stores": stores, "results": results}, ensure_ascii=False, indent=2))
    print(f"\n詳細: {path}")


if __name__ == "__main__":
    main()
