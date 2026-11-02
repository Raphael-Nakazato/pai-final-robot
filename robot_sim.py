"""
PAI 最終課題（下準備）: 言葉で指示すると、ロボットが色つきボールを順番に回るシミュレーション。

流れ
  1. 日本語の指示（例:「青のあと赤に行って、最後に緑」）を受け取る
  2. parse_command() で、行く順番の色リストに変える   ← ここを AI（LLM）にする
  3. MuJoCo の中でロボットがその順番にボールへ近づいて押す
  4. 上からのカメラ映像を mp4 に書き出す

このファイルは Colab でもローカルでも動く。LLM が使えないときは、かんたんなルールで読みとる。
"""
import json
import re

import imageio
import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# 1. シーン（床・ロボット・ボール3つ）を MJCF という形式で書く
# ---------------------------------------------------------------------------
BALLS = {
    "red":   {"pos": (0.8, 0.6),   "rgba": "0.9 0.15 0.15 1"},
    "green": {"pos": (-0.7, 0.7),  "rgba": "0.15 0.8 0.25 1"},
    "blue":  {"pos": (0.2, -0.8),  "rgba": "0.15 0.35 0.95 1"},
}

def build_xml():
    balls = "\n".join(
        f'''<body name="{c}" pos="{b['pos'][0]} {b['pos'][1]} 0.08">
              <freejoint/>
              <geom type="sphere" size="0.08" rgba="{b['rgba']}" mass="0.05"/>
            </body>'''
        for c, b in BALLS.items()
    )
    return f"""
<mujoco>
  <option timestep="0.01"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba="0.92 0.92 0.88 1"/>
    <camera name="top" pos="0 0 3.2" xyaxes="1 0 0 0 1 0"/>
    <body name="robot" pos="0 0 0.06">
      <joint name="x" type="slide" axis="1 0 0" damping="5"/>
      <joint name="y" type="slide" axis="0 1 0" damping="5"/>
      <geom type="cylinder" size="0.12 0.05" rgba="0.2 0.2 0.25 1" mass="1"/>
      <geom type="box" pos="0.1 0 0.03" size="0.03 0.03 0.02" rgba="1 0.8 0 1" mass="0.01"/>
    </body>
    {balls}
  </worldbody>
  <actuator>
    <velocity joint="x" kv="20"/>
    <velocity joint="y" kv="20"/>
  </actuator>
</mujoco>"""

# ---------------------------------------------------------------------------
# 2. 指示を読みとる（AI の部分）
# ---------------------------------------------------------------------------
COLOR_WORDS = {"red": ["赤", "あか", "red"], "green": ["緑", "みどり", "green"], "blue": ["青", "あお", "blue"]}

def parse_rule(text):
    """LLM が使えないとき用：文章の中で色が出てきた順に並べる。"""
    hits = []
    for color, words in COLOR_WORDS.items():
        for w in words:
            for m in re.finditer(re.escape(w), text):
                hits.append((m.start(), color))
    order = []
    for _, c in sorted(hits):
        if c not in order:
            order.append(c)
    return order

def parse_llm(text, generate):
    """generate(prompt) -> str を渡すと、LLM に色の順番を JSON で答えさせる。"""
    prompt = (
        "あなたはロボットの指示を読みとる係です。"
        "次の指示で、ロボットが行くボールの色を順番どおりに、"
        'JSON の配列で答えてください。使える色は "red", "green", "blue" だけです。'
        '例1: 指示「緑に行ってから赤」→ ["green", "red"]\n'
        '例2: 指示「最後に青、その前に赤に行って」→ ["red", "blue"]\n'
        '例3: 指示「赤と緑に行って。緑が先ね」→ ["green", "red"]\n'
        '例4: 指示「青、赤、緑の順で」→ ["blue", "red", "green"]\n'
        f"指示: {text}\n答え:"
    )
    out = generate(prompt)
    m = re.search(r"\[.*?\]", out, re.S)
    if not m:
        return None
    try:
        order = [c for c in json.loads(m.group(0)) if c in BALLS]
    except json.JSONDecodeError:
        return None
    return order or None

def parse_command(text, generate=None):
    if generate is not None:
        order = parse_llm(text, generate)
        if order:
            return order, "llm"
    return parse_rule(text), "rule"

# ---------------------------------------------------------------------------
# 3. ロボットを動かして動画にする
# ---------------------------------------------------------------------------
def simulate(order, fps=30, speed=0.55, max_seconds_per_target=8.0):
    """ロボットを order の順にボールへ向かわせて、フレーム（画像）のリストを返す。"""
    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    robot = model.body("robot").id
    steps_per_frame = int(round(1.0 / fps / model.opt.timestep))

    def step_and_render():
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera="top")
        frames.append(renderer.render())

    frames = []
    for _ in range(fps // 2):            # 最初に少し止まって見せる
        step_and_render()
    for color in order:
        ball = model.body(color).id
        t0 = data.time
        while data.time - t0 < max_seconds_per_target:
            diff = data.xpos[ball][:2] - data.xpos[robot][:2]
            dist = np.linalg.norm(diff)
            if dist < 0.22:              # ボールにふれたら次へ
                break
            data.ctrl[:] = speed * diff / max(dist, 1e-6)
            step_and_render()
        data.ctrl[:] = 0
        for _ in range(fps // 2):
            step_and_render()
    return frames

# ---------------------------------------------------------------------------
# 4. 文字を入れて、1本の動画にまとめる
# ---------------------------------------------------------------------------
JP = {"red": "赤", "green": "緑", "blue": "青"}

def _font(path, size):
    from PIL import ImageFont
    return ImageFont.truetype(path, size)

def _card(lines, font_path, size=(640, 480), bg=(20, 40, 60)):
    """文字だけの画面（タイトルなど）を作る。lines = [(文字, 大きさ, 色), ...]"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    total = sum(s + 14 for _, s, _ in lines)
    y = (size[1] - total) // 2
    for text, s, color in lines:
        f = _font(font_path, s)
        w = d.textlength(text, font=f)
        d.text(((size[0] - w) / 2, y), text, font=f, fill=color)
        y += s + 14
    return np.array(img)

def _caption(frame, top, bottom, font_path):
    """ロボットの映像の上に、指示と AI の読みとりを重ねる。"""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, 640, 70], fill=(0, 0, 0, 160))
    d.text((12, 8), top, font=_font(font_path, 20), fill=(255, 255, 255))
    d.text((12, 38), bottom, font=_font(font_path, 20), fill=(255, 220, 90))
    return np.array(img)

def make_video(commands, generate=None, font_path=None, out_path="final.mp4", fps=30,
               title="ことばで指示すると動くロボット", subtitle="LLM × MuJoCo シミュレーション"):
    """指示のリストから、タイトル → 各指示の実行 → おわり の動画を作る。"""
    frames = []
    hold = lambda img, sec: frames.extend([img] * int(fps * sec))

    hold(_card([(title, 34, (255, 255, 255)), (subtitle, 22, (170, 210, 255))], font_path), 4)
    results = []
    for i, cmd in enumerate(commands, 1):
        order, how = parse_command(cmd, generate)
        results.append((cmd, order, how))
        read = " → ".join(JP[c] for c in order) if order else "（読みとれず）"
        hold(_card([(f"指示 {i}", 26, (170, 210, 255)), (f"「{cmd}」", 22, (255, 255, 255))], font_path), 3)
        for f in simulate(order, fps=fps):
            frames.append(_caption(f, f"指示{i}：{cmd}", f"AI の読みとり：{read}", font_path))
    hold(_card([("おわり", 34, (255, 255, 255)), ("ご視聴ありがとうございました", 22, (170, 210, 255))], font_path), 4)

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path, len(frames) / fps, results

def run(order, out_path="robot.mp4", fps=30, max_seconds_per_target=6.0):
    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)   # 最初に位置を計算しておく（これがないと全部 0 に見える）
    renderer = mujoco.Renderer(model, 480, 640)
    robot = model.body("robot").id
    steps_per_frame = int(round(1.0 / fps / model.opt.timestep))

    frames = []
    for color in order:
        ball = model.body(color).id
        t0 = data.time
        while data.time - t0 < max_seconds_per_target:
            r = data.xpos[robot][:2]
            b = data.xpos[ball][:2]
            diff = b - r
            dist = np.linalg.norm(diff)
            if dist < 0.22:          # ボールにふれたら次へ
                break
            v = 1.2 * diff / max(dist, 1e-6)
            data.ctrl[:] = v
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="top")
            frames.append(renderer.render())
        data.ctrl[:] = 0
        for _ in range(fps // 2):      # 少し止まって見せる
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            renderer.update_scene(data, camera="top")
            frames.append(renderer.render())

    imageio.mimsave(out_path, frames, fps=fps)
    return out_path, len(frames) / fps

if __name__ == "__main__":
    cmd = "青のボールに行ってから赤、最後に緑をタッチして"
    order, how = parse_command(cmd)
    print("指示:", cmd)
    print("読みとった順番:", order, f"({how})")
    path, sec = run(order)
    print(f"動画: {path} ({sec:.1f} 秒)")
