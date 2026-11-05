# jev-playground

飲食店の売上が下がった原因を、TypeSafe AI の Jev（[typesafe-sdk](https://github.com/typesafe-ai/typesafe-sdk-python) の `jev-latest`）に聞いてみる作業場所。Jev への聞き方を段階的に変える実験と、ローカルの LLM（Ollama）との比較がある。

使っている店舗データは、正解ラベル付きのダミーデータで、実在の店舗のものではない。

## 準備

```bash
uv sync
```

```bash
cp .env.example .env
```

`.env` の `TYPESAFE_API_KEY` に、TypeSafe AI の API キーを書く。

LLM との比較もする場合は、Ollama を起動し、モデルを取得しておく（既定は `llama3:8b`）。

```bash
ollama pull llama3:8b
```

## ファイル

| ファイル | 内容 |
|---|---|
| `experiments.py` | Jev への聞き方を4段階で変えていく実験 |
| `compare.py` | Jev と LLM で、原因の判定の正解率・速度・答えのぶれ・確信度・費用を比べる |

結果は `results/` に JSON で保存される（Git 管理外）。

## experiments.py

プレイグラウンドで手作業で試していた流れをスクリプトにしたもの。

1. 答えがはっきりしたデータで聞く（Choice / Score / Noul）
2. 原因が混ざったデータで聞く（比較の基準なし → あり）
3. 原因ごとに Noul で聞く（質問に観察を混ぜる）
4. 変化率を計算して渡し、判断だけを聞く

```bash
uv run experiments.py
```

## compare.py

```bash
uv run compare.py
```

主なオプション:

| オプション | 意味 | 既定値 |
|---|---|---|
| `--only jev` / `--only llm` | 片方だけ実行する。`llm` なら API キー不要 | 両方 |
| `--llm-model` | 比べる Ollama のモデル | `llama3:8b` |
| `--per-cause` | 原因ごとの店舗数（合計はこの4倍） | 5 |
| `--repeats` | 同じ店を何回判定させるか（ぶれを見る） | 3 |
| `--limit` | 先頭から何件だけ使うか | すべて |
| `--seed` | ダミーデータの乱数シード | 42 |

少しだけ試すなら次のようにする。

```bash
uv run compare.py --limit 4 --repeats 1
```

## ライセンス

MIT
