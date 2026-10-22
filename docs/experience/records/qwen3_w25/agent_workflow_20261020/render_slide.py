#!/usr/bin/env python3
"""Rebuild one editable 16:9 slide and the matching vector PDF/PNG preview.

Run from an activated bitvae shell. All rendered elements share inch coordinates;
the PowerPoint contains native text, shapes, connectors and data markers, no
flattened screenshot. PDF/PNG use Matplotlib, not a PowerPoint rendering engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import posixpath
import zipfile

from lxml import etree as ET
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
FONT = Path("/home/star/.cache/vaellm_figure_fonts/NotoSansSC-Regular.ttf")
BOLD_FONT = Path("/home/star/.cache/vaellm_figure_fonts/NotoSansSC-Bold.ttf")
THEME_REFERENCE = Path("/home/star/.codex/plugins/cache/openai-curated-remote/openai-templates/0.1.6/skills/artifact-template-state-of-art/assets/reference.pptx")
FONT_NAME = "Noto Sans SC"
W, H = 40 / 3, 7.5
EMU = 914400
NAVY = "0A326B"
BLUE = "1467AF"
BLACK = "172231"
MUTED = "5C6877"
RED = "C93E3E"
PALE = "F1F6FB"
EDGE = "B9CFE4"
LIGHT = "DEE6EE"
WHITE = "FFFFFF"

# Edit wording and verified historical values here; no presentation service is needed.
TITLE = "VAELLM：经验驱动的自进化量化"
PROBLEM = "问题：长链路试错，结论易丢失，重复验证消耗算力"
APPROACH = "思路：目标与边界写入 AGENTS.md，让实验结果持续改进下一轮决策"
NODES = [
    {"date": "10/1", "score": 61.33014121734129, "label": "修核后纯 CAT"},
    {"date": "10/2", "score": 66.57623885036112, "label": "逐类融合恢复"},
    {"date": "10/3", "score": 67.13033274975987, "label": "CE50 / CE10\n互补组装"},
    {"date": "10/4", "score": 67.80452258209438, "label": "同源分支训练\n再组装"},
    {"date": "10/4", "score": 69.31298018402366, "label": "L0 down 编码＋\ndecoder 联合恢复"},
]
BPW = 2.4569887908
FOOTNOTE = (
    "Qwen3-8B · 固定八任务 0-shot · 全 24,742 题；252 个投影实测位宽含主码、decoder、保护、scale、索引及旋转等，不含 embedding / lm_head。\n"
    "代表性成品推进（非全部尝试、非单次训练）；多阶段组合收益，不作单因素归因；不等同通用量化或隐藏测试结果。后续九项目标另计。"
)
NOTES = """汇报口径与解释
这页展示 VAELLM 的经验和决策流程怎样迭代：证据进入外部工作树，提炼为带条件、边界和下次动作的经验，下一轮检索复用。所谓“自进化”不是 AI 自行修改自身基础模型。
人类讨论推进方案：2026-10-01 澄清“成品不留外置 LoRA”，允许训练期在尚未压缩的 dense 类别使用临时 remaining LoRA 并融合；允许模型级 decoder/码位蒸馏。因此本页仅表述“成品无外置 LoRA”，不声称全流程从未使用 LoRA。

历史目标与五个节点
这是 2026-10-01 至 10-04 原“投影位宽不超过 2.5 bit、固定八任务均分达到 69%”阶段，后续 MMLU、九项目标和 Llama 任务另计。
五点为不同代表性成品的关键有效节点，非单次训练收敛曲线；省略了失败、回落或无收益的尝试，不能读成每一轮都会上涨。
修核后纯 CAT 61.33014121734129%；逐类融合恢复 66.5762388504%；CE50 attention + CE10 MLP 组合 67.13033275%；同源分支训练后组合 67.8045225821%；L0 down 联合恢复第 1200 步 69.31298018402366%。
首点使用 eval_fixed_decoder；旧解码核的 61.320222% 不采用。前述五点为相同评测口径、相同实际投影位宽、rank0 成品；历史分支中间训练允许临时 LoRA，最终融合并无外置分支。
最终相对首点提高 7.98283896668237 个百分点，页面显示 +7.983 个百分点。这是多阶段组合后的观察收益，不能归因于 AGENTS.md、某一单因素或某种因果效率提升。

完整评测协议
模型 Qwen/Qwen3-8B，revision b968826d9c46dd6066d109eabc6255188de91218。lm-eval 0.4.8，固定种子的历史单次运行；全部任务 0-shot、无 limit、不套聊天模板，标准候选因果似然。
BoolQ / RTE / Winogrande / MMLU 使用 acc,none；ARC-Easy / ARC-Challenge / OpenBookQA / PIQA 使用 acc_norm,none。MMLU 先按 57 学科共 14,042 题合成一项，再与另外七项等权平均；总题数 24,742。MMLU 实际 batch64，其他七项 batch32；首点 CLI 虽为 auto，实际日志也是 64/32。
fast tokenizer Qwen2TokenizerFast；v6 strict 加载，BF16 解码缓存，未启用 HiFloat4 激活量化。成品八项均分 69.31298018402366%，MMLU 61.0240706452%；均分达标不代表每项均达 69%。成品未做 PPL，无长上下文、生成质量或吞吐结论。
2.4569887908 bit/weight 来自实际张量总位数 17,065,672,704 / 252 个原始投影权重 6,945,767,424；包含主编码、decoder、INT8 保护权重、scale、通道索引及旋转符号等，不含 embedding、lm_head、原有 norm。该值不是完整模型每参数位宽。
历史固定八任务被多次用于方案与保存点选择，独立验收是重新加载成品后完整复评，并非隐藏独立测试。现有材料没有多种子误差条或显著性检验。训练使用任务 train/auxiliary_train，未从正式 validation/test 分区采训练样本；此审计不等价于全互联网语料去污染。

正负经验与复用
负例：从 66.57623885% 起点按原配方追加 decoder 2000 步，末点 66.16179538%，五个保存点均未超过起点；该原配方退役并保留最小证据。仅适用于该配方和条件，不能泛化为所有 decoder 恢复无效。
具体复用：10/15 新 CAT 沿用教师 RoPE 修复，仅补加载与设备迁移检查。这是复用已有证据的事实，不是下游性能收益。已有证据仍适用才直接复用；路径、资源条件变化或出现实质异常时，只补验受影响部分。

主要来源
AGENTS.md：目标、权限、边界、流程、最小充分验证和收尾规则。
docs/experience/records/qwen3_w25/2026-10-04_qwen3_8b_69_report.md
docs/guides/qwen3_8b_69_reproduction.md
docs/experience/lessons/memory_precision.md（教师 RoPE 修复及复用条件）
docs/experience/records/w25_transfer/2026-10-04_mmlu65_llama31.md（2026-10-15 新 CAT 复用事实）
docs/experience/records/qwen3_w25/2026-09-30_goal.md：51 行人类澄清，539 行修核起点，741/946 行逐类恢复及同批复评，2438 行互补组合，3484 行格式×训练配对，3524 行二次组合，3641 行最终达标。
docs/guides/qwen3_8b_69_reproduction.md：84 行起将达标来源链固化为执行流程。
五节点原始验收（项目相对路径；相同目录的 mmlu.json / seven.json 为原始完整评测，已分别复算核对）：
result/qwen3_8b_w25_20260930/reconstruction_block32_o3_down3/eval_fixed_decoder/acceptance.json
result/qwen3_8b_w25_20260930/remaining_prefix_block32_s500/eval_batch64_32/acceptance.json
result/qwen3_8b_w25_20260930/decoder_ce50_ce10_mlp/eval/acceptance.json
result/qwen3_8b_w25_20260930/assembled_ce50_chatml_joint_bits00_s2000_attn/eval/acceptance.json
result/qwen3_8b_w25_20260930/e2e_best_combo_down0_joint_s2000/step_1200/acceptance.json
第四点把来自同一 67.1303327498% 起点的 CE50 ChatML 局部 joint 分支，与全 decoder KL 分支的 attention 组合；不是随意拼接两个模型。

生成与验证说明
本 PPTX 的文本、方框、折线、连接线、箭头和数据节点均为原生可编辑对象；无截图替代正文。中文字体设为 Noto Sans SC，查看端须可用此字体以保持布局。
PDF 与 PNG 从同一几何和文案生成，PDF 保留矢量线条及嵌入字体。已做 XML、关系、坐标、字体覆盖和文本边界检查，并人工检查 PNG；未使用真实 PowerPoint / LibreOffice 渲染，因此不同 Office 渲染引擎仍可能有字体回退或换行差异。
"""


@dataclass
class Item:
    kind: str
    name: str
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    text: str = ""
    size: float = 14.0
    color: str = BLACK
    bold: bool = False
    align: str = "left"
    fill: str | None = None
    edge: str | None = None
    lw: float = 1.0
    points: tuple[tuple[float, float], ...] = ()
    dashed: bool = False


class Slide:
    def __init__(self):
        self.items: list[Item] = []

    def text(self, name, x, y, w, h, text, size=14, color=BLACK, bold=False, align="left"):
        self.items.append(Item("text", name, x, y, w, h, text, size, color, bold, align))

    def rect(self, name, x, y, w, h, fill=None, edge=None, lw=1):
        self.items.append(Item("rect", name, x, y, w, h, fill=fill, edge=edge, lw=lw))

    def ellipse(self, name, x, y, w, h, fill, edge=None, lw=1):
        self.items.append(Item("ellipse", name, x, y, w, h, fill=fill, edge=edge, lw=lw))

    def line(self, name, points, color=BLUE, lw=1.2, dashed=False):
        self.items.append(Item("line", name, color=color, lw=lw, points=tuple(points), dashed=dashed))

    def polygon(self, name, points, fill, edge=None, lw=0.8):
        self.items.append(Item("polygon", name, fill=fill, edge=edge, lw=lw, points=tuple(points)))

    def arrow(self, name, points, color=BLUE, lw=1.25, head=0.075):
        self.line(name, points, color, lw)
        end, prev = points[-1], points[-2]
        dx, dy = end[0] - prev[0], end[1] - prev[1]
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        base = (end[0] - head * ux, end[1] - head * uy)
        self.polygon(name + " arrowhead", [end, (base[0] - head * uy * 0.45, base[1] + head * ux * 0.45),
                                            (base[0] + head * uy * 0.45, base[1] - head * ux * 0.45)], color)


def layout() -> Slide:
    s = Slide()
    # Academic house style: white page, double chevron, navy title and bottom rule.
    for x in (0.36, 0.51):
        s.polygon("title chevron", [(x, 0.39), (x + .105, .51), (x, .63), (x + .065, .63),
                                   (x + .17, .51), (x + .065, .39)], NAVY)
    s.text("title", .84, .27, 12.0, .58, TITLE, 31.3, NAVY, True)
    s.text("problem", .56, 1.00, 12.15, .29, PROBLEM, 16.2)
    s.text("approach", .56, 1.37, 12.15, .29, APPROACH, 16.2)
    s.line("intro divider", [(.55, 1.83), (12.79, 1.83)], LIGHT, .85)

    s.text("workflow heading", .56, 1.96, 8.45, .31, "证据 → 经验 → 下一轮决策", 17.5, BLUE, True)
    s.text("role heading", 9.57, 1.96, 3.13, .31, "人机协作与边界", 17.5, BLUE, True)
    s.line("role divider", [(9.29, 1.98), (9.29, 4.09)], EDGE, 1.0)

    labels = ["检索经验", "设计实验", "最小充分验证", "后台执行", "完整评测"]
    for i, label in enumerate(labels):
        x = .63 + i * 1.70
        s.rect(f"workflow step {i + 1}", x, 2.40, 1.43, .55, PALE, EDGE, 1.0)
        s.text(f"workflow step {i + 1} label", x + .025, 2.43, 1.38, .48, label, 14.2, BLUE, True, "center")
        if i < 4:
            s.arrow(f"workflow link {i + 1}", [(x + 1.47, 2.675), (x + 1.64, 2.675)], BLUE, 1.25, .065)
    s.arrow("evaluation to knowledge", [(8.145, 2.97), (8.145, 3.10)], BLUE, 1.05, .055)

    s.rect("knowledge band", .63, 3.13, 8.23, .55, PALE)
    memories = [(0.83, 2.25, "下一轮决策\n检索并复用"),
                (3.55, 2.80, "lessons/\n提炼条件、边界与下次动作"),
                (6.71, 2.10, "records/ ＋ 结果目录\n记录正负结果与原始证据")]
    for i, (x, w, label) in enumerate(memories):
        s.text(f"knowledge memory {i + 1}", x, 3.15, w, .51, label, 12.5, BLUE, False, "center")
    s.arrow("memory link 1", [(3.47, 3.405), (3.20, 3.405)], BLUE, 1.1, .065)
    s.arrow("memory link 2", [(6.66, 3.405), (6.42, 3.405)], BLUE, 1.1, .065)
    s.arrow("knowledge feedback", [(.65, 3.405), (.39, 3.405), (.39, 2.675), (.61, 2.675)], BLUE, 1.05, .065)
    s.text("reuse rule", .63, 3.79, 8.37, .23,
           "已有证据仍适用：直接复用；路径 / 资源变化或实质异常：仅补验受影响部分", 11.4, MUTED)
    s.text("reuse example", .63, 4.07, 8.37, .22,
           "复用实例：10/15 新 CAT 沿用教师 RoPE 修复，仅补加载与设备迁移检查", 10.7, BLUE)

    s.text("agents memory", 9.57, 2.37, 3.10, .26, "AGENTS.md：目标 / 权限 / 流程", 12.1, BLUE)
    s.text("AI role", 9.57, 2.80, 3.10, .48, "AI 自主：查证、实现、调参\n并行实验、评测与归档", 13.3)
    s.text("human role", 9.57, 3.40, 3.10, .48, "人类掌舵：目标验收、资源授权\n关键取舍、实质歧义", 12.7)
    s.text("confirm boundary", 9.57, 4.00, 3.10, .26, "边界变更 → 先确认", 13.1, RED, True)

    s.line("history divider", [(.55, 4.46), (12.79, 4.46)], LIGHT, .85)
    s.text("history heading", .56, 4.59, 8.50, .31, "真实历史｜10/1—10/4 原 69% 目标阶段", 17.5, BLUE, True)
    delta = NODES[-1]["score"] - NODES[0]["score"]
    s.text("gain", 9.57, 4.57, 3.11, .35, f"＋{delta:.3f} 个百分点", 18.7, RED, True)
    s.text("human decision", .63, 5.01, 8.45, .27,
           "10/1 讨论澄清：成品不留外置 LoRA，允许训练期临时融合 → 恢复路线推进", 11.6)
    s.text("storage and lora", 9.57, 5.01, 3.10, .27,
           "2.457 bit/weight · 成品无外置 LoRA", 10.3, BLUE)

    xs = [1.30, 3.85, 6.40, 8.95, 11.50]
    base_y = 6.17
    yscale = .1035
    pts = [(x, base_y - (n["score"] - NODES[0]["score"]) * yscale) for x, n in zip(xs, NODES)]
    target_y = base_y - (69 - NODES[0]["score"]) * yscale
    s.line("69 percent target", [(.76, target_y), (12.67, target_y)], RED, .95, True)
    s.rect("target label background", 11.94, target_y - .14, .79, .25, WHITE)
    s.text("target label", 11.95, target_y - .14, .78, .25, "目标 69%", 10.4, RED, False, "right")
    s.line("historical progression", pts, BLUE, 2.25)
    for i, (node, (x, y)) in enumerate(zip(NODES, pts)):
        color = RED if i == len(NODES) - 1 else BLUE
        if y < 6.18:
            s.line(f"data stem {i + 1}", [(x, y + .075), (x, 6.25)], LIGHT, .75, True)
        s.ellipse(f"history point {i + 1}", x - .055, y - .055, .11, .11, color, WHITE, .7)
        s.text(f"history score {i + 1}", x - .98, 6.32, 1.96, .30, f"{node['score']:.3f}%", 16.8, color, True, "center")
        s.text(f"history action {i + 1}", x - 1.15, 6.67, 2.30, .42, node["date"] + "  " + node["label"], 11.4, BLACK, False, "center")

    s.text("protocol footnote", .56, 7.17, 12.18, .285, FOOTNOTE, 9.0, MUTED)
    s.rect("bottom navy rule", 0, 7.473, W, .027, NAVY)
    return s


NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
      "ct": "http://schemas.openxmlformats.org/package/2006/content-types"}


def sub(parent, tag, attrs=None, text=None):
    prefix, local = tag.split(":")
    el = ET.SubElement(parent, f"{{{NS[prefix]}}}{local}", attrs or {})
    if text is not None:
        el.text = text
    return el


def xml(root):
    return ET.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def qroot(tag, attrs=None):
    p, local = tag.split(":")
    return ET.Element(f"{{{NS[p]}}}{local}", attrs or {}, nsmap={k: NS[k] for k in ("a", "r", "p")})


def unit(value):
    return str(round(value * EMU))


def solid(parent, color):
    fill = sub(parent, "a:solidFill")
    sub(fill, "a:srgbClr", {"val": color})


def grp_tree(parent):
    tree = sub(parent, "p:spTree")
    nv = sub(tree, "p:nvGrpSpPr")
    sub(nv, "p:cNvPr", {"id": "1", "name": ""})
    sub(nv, "p:cNvGrpSpPr")
    sub(nv, "p:nvPr")
    sppr = sub(tree, "p:grpSpPr")
    xfrm = sub(sppr, "a:xfrm")
    for name in ("off", "chOff"):
        sub(xfrm, "a:" + name, {"x": "0", "y": "0"})
    for name in ("ext", "chExt"):
        sub(xfrm, "a:" + name, {"cx": "0", "cy": "0"})
    return tree


def native_shape(tree, item, ident):
    connector = item.kind == "line"
    sp = sub(tree, "p:cxnSp" if connector else "p:sp")
    nv = sub(sp, "p:nvCxnSpPr" if connector else "p:nvSpPr")
    sub(nv, "p:cNvPr", {"id": str(ident), "name": f"{ident:03d} {item.name}"})
    sub(nv, "p:cNvCxnSpPr" if connector else "p:cNvSpPr", {"txBox": "1"} if item.kind == "text" else {})
    sub(nv, "p:nvPr")
    sppr = sub(sp, "p:spPr")
    if item.points:
        x = min(p[0] for p in item.points)
        y = min(p[1] for p in item.points)
        w = max(max(p[0] for p in item.points) - x, 1 / EMU)
        h = max(max(p[1] for p in item.points) - y, 1 / EMU)
    else:
        x, y, w, h = item.x, item.y, item.w, item.h
    xf = sub(sppr, "a:xfrm")
    sub(xf, "a:off", {"x": unit(x), "y": unit(y)})
    sub(xf, "a:ext", {"cx": unit(w), "cy": unit(h)})
    if item.points:
        geom = sub(sppr, "a:custGeom")
        for tag in ("avLst", "gdLst", "ahLst", "cxnLst"):
            sub(geom, "a:" + tag)
        sub(geom, "a:rect", {"l": "0", "t": "0", "r": "r", "b": "b"})
        paths = sub(geom, "a:pathLst")
        path = sub(paths, "a:path", {"w": "100000", "h": "100000", "fill": "none" if connector else "norm"})
        for i, (px, py) in enumerate(item.points):
            cmd = sub(path, "a:moveTo" if i == 0 else "a:lnTo")
            sub(cmd, "a:pt", {"x": str(round((px - x) / w * 100000)), "y": str(round((py - y) / h * 100000))})
        if item.kind == "polygon":
            sub(path, "a:close")
    else:
        geom = sub(sppr, "a:prstGeom", {"prst": "ellipse" if item.kind == "ellipse" else "rect"})
        sub(geom, "a:avLst")
    if item.fill:
        solid(sppr, item.fill)
    else:
        sub(sppr, "a:noFill")
    ln = sub(sppr, "a:ln", {"w": str(round(item.lw * 12700)), "cap": "flat", "cmpd": "sng", "algn": "ctr"})
    if connector or item.edge:
        solid(ln, item.color if connector else item.edge)
        sub(ln, "a:prstDash", {"val": "dash" if item.dashed else "solid"})
        sub(ln, "a:round")
    else:
        sub(ln, "a:noFill")
    if item.kind == "text":
        tx = sub(sp, "p:txBody")
        body = sub(tx, "a:bodyPr", {"wrap": "none", "lIns": "0", "tIns": "0", "rIns": "0", "bIns": "0", "anchor": "ctr", "rtlCol": "0"})
        sub(body, "a:noAutofit")
        sub(tx, "a:lstStyle")
        for line in item.text.split("\n"):
            p = sub(tx, "a:p")
            ppr = sub(p, "a:pPr", {"algn": {"left": "l", "center": "ctr", "right": "r"}[item.align], "marL": "0", "marR": "0", "indent": "0"})
            spacing = sub(ppr, "a:lnSpc")
            sub(spacing, "a:spcPts", {"val": str(round(item.size * 116))})
            for tag in ("spcBef", "spcAft"):
                spacing = sub(ppr, "a:" + tag)
                sub(spacing, "a:spcPts", {"val": "0"})
            sub(ppr, "a:buNone")
            r = sub(p, "a:r")
            rpr = sub(r, "a:rPr", {"lang": "zh-CN", "altLang": "en-US", "sz": str(round(item.size * 100)), "b": "1" if item.bold else "0", "kern": "0", "spc": "0", "dirty": "0"})
            solid(rpr, item.color)
            for tag in ("latin", "ea", "cs"):
                sub(rpr, "a:" + tag, {"typeface": FONT_NAME})
            sub(r, "a:t", text=line)
            sub(p, "a:endParaRPr", {"lang": "zh-CN", "sz": str(round(item.size * 100))})


def rels(entries):
    root = ET.Element(f"{{{NS['rel']}}}Relationships", nsmap={None: NS["rel"]})
    for ident, kind, target in entries:
        ET.SubElement(root, f"{{{NS['rel']}}}Relationship", {"Id": ident, "Type": NS["r"] + "/" + kind, "Target": target})
    return xml(root)


def clr_map(parent):
    sub(parent, "p:clrMap", {"bg1": "lt1", "tx1": "dk1", "bg2": "lt2", "tx2": "dk2", **{x: x for x in ["accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"]}})


def write_pptx(slide):
    parts = {}
    root = qroot("p:sld")
    cs = sub(root, "p:cSld", {"name": TITLE})
    bg = sub(cs, "p:bg")
    bgpr = sub(bg, "p:bgPr")
    solid(bgpr, WHITE)
    sub(bgpr, "a:effectLst")
    tree = grp_tree(cs)
    for ident, item in enumerate(slide.items, 2):
        native_shape(tree, item, ident)
    sub(sub(root, "p:clrMapOvr"), "a:masterClrMapping")
    parts["ppt/slides/slide1.xml"] = xml(root)
    parts["ppt/slides/_rels/slide1.xml.rels"] = rels([("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "notesSlide", "../notesSlides/notesSlide1.xml")])

    layout_root = qroot("p:sldLayout", {"type": "blank", "preserve": "1"})
    grp_tree(sub(layout_root, "p:cSld", {"name": "Blank"}))
    sub(sub(layout_root, "p:clrMapOvr"), "a:masterClrMapping")
    parts["ppt/slideLayouts/slideLayout1.xml"] = xml(layout_root)
    parts["ppt/slideLayouts/_rels/slideLayout1.xml.rels"] = rels([("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")])

    master = qroot("p:sldMaster")
    grp_tree(sub(master, "p:cSld", {"name": "VAELLM Academic"}))
    clr_map(master)
    ids = sub(master, "p:sldLayoutIdLst")
    sub(ids, "p:sldLayoutId", {"id": "2147483649", f"{{{NS['r']}}}id": "rId1"})
    tx = sub(master, "p:txStyles")
    for style in ("titleStyle", "bodyStyle", "otherStyle"):
        el = sub(tx, "p:" + style)
        level = sub(el, "a:lvl1pPr", {"algn": "l"})
        rpr = sub(level, "a:defRPr", {"sz": "1400"})
        solid(rpr, BLACK)
        for tag in ("latin", "ea", "cs"):
            sub(rpr, "a:" + tag, {"typeface": FONT_NAME})
    parts["ppt/slideMasters/slideMaster1.xml"] = xml(master)
    parts["ppt/slideMasters/_rels/slideMaster1.xml.rels"] = rels([("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"), ("rId2", "theme", "../theme/theme1.xml")])

    with zipfile.ZipFile(THEME_REFERENCE) as z:
        theme_names = sorted(n for n in z.namelist() if n.startswith("ppt/theme/theme") and n.endswith(".xml"))
        theme = ET.fromstring(z.read(theme_names[0]))
    theme.set("name", "VAELLM Academic")
    scheme = theme.find("a:themeElements/a:fontScheme", NS)
    scheme.set("name", "Noto Sans SC")
    for fonts in scheme:
        for elem in fonts:
            if ET.QName(elem).localname in {"latin", "ea", "cs", "font"}:
                elem.set("typeface", FONT_NAME)
    parts["ppt/theme/theme1.xml"] = xml(theme)

    notesmaster = qroot("p:notesMaster")
    grp_tree(sub(notesmaster, "p:cSld"))
    clr_map(notesmaster)
    sub(notesmaster, "p:notesStyle")
    parts["ppt/notesMasters/notesMaster1.xml"] = xml(notesmaster)
    parts["ppt/notesMasters/_rels/notesMaster1.xml.rels"] = rels([("rId1", "theme", "../theme/theme1.xml")])
    notes = qroot("p:notes")
    ntree = grp_tree(sub(notes, "p:cSld"))
    sp = sub(ntree, "p:sp")
    nv = sub(sp, "p:nvSpPr")
    sub(nv, "p:cNvPr", {"id": "2", "name": "Speaker Notes"})
    sub(nv, "p:cNvSpPr")
    nvp = sub(nv, "p:nvPr")
    sub(nvp, "p:ph", {"type": "body", "idx": "1"})
    sub(sp, "p:spPr")
    body = sub(sp, "p:txBody")
    sub(body, "a:bodyPr")
    sub(body, "a:lstStyle")
    for line in NOTES.splitlines():
        p = sub(body, "a:p")
        r = sub(p, "a:r")
        rpr = sub(r, "a:rPr", {"lang": "zh-CN", "sz": "1100"})
        sub(rpr, "a:ea", {"typeface": FONT_NAME})
        sub(r, "a:t", text=line)
    sub(sub(notes, "p:clrMapOvr"), "a:masterClrMapping")
    parts["ppt/notesSlides/notesSlide1.xml"] = xml(notes)
    parts["ppt/notesSlides/_rels/notesSlide1.xml.rels"] = rels([("rId1", "notesMaster", "../notesMasters/notesMaster1.xml"), ("rId2", "slide", "../slides/slide1.xml")])

    pres = qroot("p:presentation", {"saveSubsetFonts": "1", "autoCompressPictures": "0"})
    ids = sub(pres, "p:sldMasterIdLst")
    sub(ids, "p:sldMasterId", {"id": "2147483648", f"{{{NS['r']}}}id": "rId1"})
    ids = sub(pres, "p:notesMasterIdLst")
    sub(ids, "p:notesMasterId", {f"{{{NS['r']}}}id": "rId3"})
    ids = sub(pres, "p:sldIdLst")
    sub(ids, "p:sldId", {"id": "256", f"{{{NS['r']}}}id": "rId2"})
    sub(pres, "p:sldSz", {"cx": unit(W), "cy": unit(H), "type": "screen16x9"})
    sub(pres, "p:notesSz", {"cx": "6858000", "cy": "9144000"})
    sub(pres, "p:defaultTextStyle")
    parts["ppt/presentation.xml"] = xml(pres)
    parts["ppt/_rels/presentation.xml.rels"] = rels([("rId1", "slideMaster", "slideMasters/slideMaster1.xml"), ("rId2", "slide", "slides/slide1.xml"), ("rId3", "notesMaster", "notesMasters/notesMaster1.xml")])
    parts["_rels/.rels"] = rels([("rId1", "officeDocument", "ppt/presentation.xml")])

    ct = ET.Element(f"{{{NS['ct']}}}Types", nsmap={None: NS["ct"]})
    for ext, mime in [("rels", "application/vnd.openxmlformats-package.relationships+xml"), ("xml", "application/xml")]:
        ET.SubElement(ct, f"{{{NS['ct']}}}Default", {"Extension": ext, "ContentType": mime})
    overrides = {"presentation": "presentation.main", "slides/slide1": "slide", "slideLayouts/slideLayout1": "slideLayout", "slideMasters/slideMaster1": "slideMaster", "notesSlides/notesSlide1": "notesSlide", "notesMasters/notesMaster1": "notesMaster"}
    for path, kind in overrides.items():
        ET.SubElement(ct, f"{{{NS['ct']}}}Override", {"PartName": "/ppt/" + path + ".xml", "ContentType": "application/vnd.openxmlformats-officedocument.presentationml." + kind + "+xml"})
    ET.SubElement(ct, f"{{{NS['ct']}}}Override", {"PartName": "/ppt/theme/theme1.xml", "ContentType": "application/vnd.openxmlformats-officedocument.theme+xml"})
    parts["[Content_Types].xml"] = xml(ct)
    path = OUT / "agent_workflow.pptx"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return path


def render_and_check(slide):
    for path in (FONT, BOLD_FONT):
        if not path.is_file():
            raise FileNotFoundError(path)
        font_manager.fontManager.addfont(str(path))
    regular = font_manager.FontProperties(fname=str(FONT))
    bold = font_manager.FontProperties(fname=str(BOLD_FONT))
    glyph_maps = {False: FT2Font(str(FONT)).get_charmap(), True: FT2Font(str(BOLD_FONT)).get_charmap()}
    for item in slide.items:
        if item.kind == "text":
            missing = {c for c in item.text if c not in "\n\t" and ord(c) not in glyph_maps[item.bold]}
            if missing:
                raise ValueError(f"Missing glyphs in {item.name}: {missing}")
    fig = plt.figure(figsize=(W, H), dpi=180, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
    ax.axis("off")
    texts = []
    for item in slide.items:
        if item.kind == "text":
            xpos = item.x + {"left": 0, "center": item.w / 2, "right": item.w}[item.align]
            art = ax.text(xpos, item.y + item.h / 2, item.text, fontsize=item.size, fontproperties=bold if item.bold else regular,
                          color="#" + item.color, ha=item.align, va="center", linespacing=1.16, parse_math=False)
            texts.append((item, art))
        elif item.kind == "rect":
            ax.add_patch(Rectangle((item.x, item.y), item.w, item.h, facecolor="#" + item.fill if item.fill else "none",
                                   edgecolor="#" + item.edge if item.edge else "none", linewidth=item.lw))
        elif item.kind == "ellipse":
            ax.add_patch(Ellipse((item.x + item.w / 2, item.y + item.h / 2), item.w, item.h,
                                facecolor="#" + item.fill, edgecolor="#" + item.edge if item.edge else "none", linewidth=item.lw))
        elif item.kind == "polygon":
            ax.add_patch(Polygon(item.points, closed=True, facecolor="#" + item.fill,
                                 edgecolor="#" + item.edge if item.edge else "none", linewidth=item.lw))
        elif item.kind == "line":
            xs, ys = zip(*item.points)
            ax.plot(xs, ys, color="#" + item.color, linewidth=item.lw, linestyle=(0, (4, 3)) if item.dashed else "-", solid_capstyle="butt")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rects = []
    errors = []
    for item, art in texts:
        bb = art.get_window_extent(renderer)
        inv = ax.transData.inverted()
        p0, p1 = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        actual = (p0[0], p1[1], p1[0], p0[1])
        width, height = actual[2] - actual[0], actual[3] - actual[1]
        if width > item.w + .014 or height > item.h + .014:
            errors.append(f"Text exceeds box: {item.name}: actual {width:.3f} x {height:.3f}, box {item.w:.3f} x {item.h:.3f}")
        if actual[0] < -.001 or actual[1] < -.001 or actual[2] > W + .001 or actual[3] > H + .001:
            errors.append(f"Text outside slide: {item.name}")
        rects.append((item.name, actual))
    for i, (name1, a) in enumerate(rects):
        for name2, b in rects[i + 1:]:
            overlap_w = min(a[2], b[2]) - max(a[0], b[0])
            overlap_h = min(a[3], b[3]) - max(a[1], b[1])
            if overlap_w > .005 and overlap_h > .005:
                errors.append(f"Text overlap: {name1} / {name2}")
    if errors:
        raise ValueError("\n".join(errors))
    fig.savefig(OUT / "agent_workflow.pdf", dpi=180, facecolor="white", metadata={"Title": TITLE, "Subject": "Verified historical workflow, editable PPTX companion", "Creator": "VAELLM render_slide.py"})
    fig.savefig(OUT / "agent_workflow.png", dpi=240, facecolor="white")
    plt.close(fig)
    with Image.open(OUT / "agent_workflow.png") as preview:
        if preview.size != (3200, 1800):
            raise ValueError(f"Unexpected PNG size: {preview.size}")
        preview.verify()
    pdf = (OUT / "agent_workflow.pdf").read_bytes()
    if b"/FontFile2" not in pdf or b"/Subtype /Image" in pdf:
        raise ValueError("PDF must contain embedded fonts and vector content")
    return {"text_boxes": len(texts), "text_boundary_and_pairwise_overlap_checks": "passed", "font_coverage": "passed", "pdf": "vector paths with embedded TrueType font; no raster images", "png": "3200 x 1800, 16:9", "preview_renderer": "Matplotlib Agg; not PowerPoint"}


def validate_pptx(path, slide):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt ZIP package")
        names = set(z.namelist())
        for name in names:
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(z.read(name))
            if name.endswith(".rels"):
                rel_root = ET.fromstring(z.read(name))
                source_dir = "" if name == "_rels/.rels" else posixpath.dirname(posixpath.dirname(name))
                ids = []
                for rel in rel_root:
                    ids.append(rel.get("Id"))
                    if rel.get("TargetMode") == "External":
                        raise ValueError(f"Unexpected external relationship in {name}")
                    target = posixpath.normpath(posixpath.join(source_dir, rel.get("Target")))
                    if target not in names:
                        raise ValueError(f"Broken relationship: {name} -> {target}")
                if len(ids) != len(set(ids)):
                    raise ValueError(f"Duplicate relationship ID: {name}")
        actual = ET.fromstring(z.read("ppt/slides/slide1.xml"))
        shapes = actual.xpath("//p:spTree/p:sp | //p:spTree/p:cxnSp", namespaces=NS)
        if len(shapes) != len(slide.items):
            raise ValueError("Native shape count differs from layout")
        actual_text = actual.xpath("//a:t/text()", namespaces=NS)
        expected_text = [line for item in slide.items if item.kind == "text" for line in item.text.split("\n")]
        if actual_text != expected_text:
            raise ValueError("PPTX text differs from shared layout")
        ids = actual.xpath("//p:cNvPr/@id", namespaces=NS)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate native shape IDs")
        for transform in actual.xpath("//p:spPr/a:xfrm", namespaces=NS):
            off, ext = transform.find("a:off", NS), transform.find("a:ext", NS)
            x, y = int(off.get("x")), int(off.get("y"))
            w, h = int(ext.get("cx")), int(ext.get("cy"))
            if min(x, y, w, h) < 0 or x + w > round(W * EMU) + 1 or y + h > round(H * EMU) + 1:
                raise ValueError("Native shape outside slide")
        if actual.xpath("//p:pic", namespaces=NS) or any(n.startswith("ppt/media/") for n in names):
            raise ValueError("Slide unexpectedly contains raster content")
        note_text = "\n".join(ET.fromstring(z.read("ppt/notesSlides/notesSlide1.xml")).xpath("//a:t/text()", namespaces=NS))
        for phrase in ("隐藏独立测试", "单因素", "临时", "PowerPoint", "memory_precision.md"):
            if phrase not in note_text:
                raise ValueError(f"Required speaker note missing: {phrase}")
    return {"package_xml_and_relationships": "passed", "native_objects": len(slide.items), "raster_images": 0, "editable_text_and_coordinates": "passed", "speaker_notes": "present"}


def main():
    slide = layout()
    render_report = render_and_check(slide)
    pptx = write_pptx(slide)
    report = {"status": "passed", "generated_utc": datetime.now(timezone.utc).isoformat(),
              "format": "16:9, one slide, native editable objects", **render_report, **validate_pptx(pptx, slide),
              "score_gain_percentage_points": NODES[-1]["score"] - NODES[0]["score"],
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p.read_bytes()).hexdigest()}
                        for p in [pptx, OUT / "agent_workflow.pdf", OUT / "agent_workflow.png"]}}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
