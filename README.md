# ことばで指示すると動くロボット（PAI 最終課題）

日本語で指示すると、LLM（大規模言語モデル）が「どのボールに、どの順番で行くか」を読みとり、
MuJoCo のシミュレーションの中でロボットがその順番どおりに動きます。

- 動画：https://youtu.be/ESJO6uD7rIY
- 使ったもの：Python / MuJoCo / Hugging Face Transformers（Qwen2.5-Instruct）/ Google Colab

## しくみ

```
日本語の指示 ──▶ LLM（Qwen2.5-3B-Instruct）──▶ 行く順番 ["blue", "red", "green"]
                                                   │
                                                   ▼
                              MuJoCo のロボットが順番にボールへ向かう ──▶ 動画（mp4）
```

- `parse_command()`：指示を LLM に渡して、色の順番を JSON の配列で答えさせる。読みとれないときは、かんたんなルールで読む
- `simulate()`：ロボットを順番どおりにボールへ向かわせて、上からのカメラ映像を集める
- `make_video()`：タイトル → 各指示の実行（指示と AI の読みとりを文字で重ねる）→ おわり、を1本の動画にする

AI は**推論だけ**に使っていて、学習はしていません。

## 実験：モデルの大きさと、プロンプトのお手本（few-shot）

同じ5問の指示で、正しい順番を読みとれた数をくらべました。

| モデル | お手本なし | お手本あり |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 1/5 | 3/5 |
| Qwen2.5-1.5B-Instruct | 2/5 | 4/5 |
| Qwen2.5-3B-Instruct | 4/5 | **5/5** |
| Qwen2.5-7B-Instruct | 4/5 | **5/5** |

- モデルを大きくするほど正解はふえたが、3B と 7B は同じ点で、まちがえる問題がちがった
- プロンプトにお手本を4つ入れると、どの大きさでも1〜2問ふえた。小さいモデルほど伸びが大きかった
- お手本ありなら 3B で 7B と同じ点になったので、軽い 3B を本番に使った
- お手本はテストと同じ文にしていない（形は似ているが色を変えた）。まったく新しい言い方でどうなるかは、今後ためしたい

## 動かし方（Google Colab）

1. `pai_final.ipynb` を Colab で開き、ランタイムを GPU（T4 以上。7B もためすなら L4 / A100）にする
2. 上のセルから順に実行し、`robot_sim.py` をアップロードする
3. 最後のセルで `final.mp4` ができる

ローカルで動きだけ確かめるときは：

```bash
pip install mujoco imageio imageio-ffmpeg numpy pillow
python robot_sim.py
```

## ファイル

- `robot_sim.py`：シミュレーション・指示の読みとり・動画づくり
- `pai_final.ipynb`：Colab 用のノートブック（実験と動画づくり）
- `results/`：実験のスクリーンショット
