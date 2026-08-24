#!/usr/bin/env python3
"""Build a Word manuscript containing the abstract and experimental results."""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

ET.register_namespace("w", W)
ET.register_namespace("r", R)
ET.register_namespace("cp", CP)
ET.register_namespace("dc", DC)
ET.register_namespace("dcterms", DCTERMS)
ET.register_namespace("xsi", XSI)
ET.register_namespace("ep", EP)
ET.register_namespace("vt", VT)


def qn(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_run_format(
    run: ET.Element,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    size: int = 20,
) -> None:
    properties = ET.SubElement(run, qn(W, "rPr"))
    fonts = ET.SubElement(properties, qn(W, "rFonts"))
    for attribute in ("ascii", "hAnsi", "cs"):
        fonts.set(qn(W, attribute), "Times New Roman")
    fonts.set(qn(W, "eastAsia"), "SimSun")
    if bold:
        ET.SubElement(properties, qn(W, "b"))
    if italic:
        ET.SubElement(properties, qn(W, "i"))
    if underline:
        underline_element = ET.SubElement(properties, qn(W, "u"))
        underline_element.set(qn(W, "val"), "single")
    size_element = ET.SubElement(properties, qn(W, "sz"))
    size_element.set(qn(W, "val"), str(size))
    size_cs = ET.SubElement(properties, qn(W, "szCs"))
    size_cs.set(qn(W, "val"), str(size))


def add_run(
    paragraph: ET.Element,
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    underline: bool = False,
    size: int = 20,
) -> ET.Element:
    run = ET.SubElement(paragraph, qn(W, "r"))
    set_run_format(
        run,
        bold=bold,
        italic=italic,
        underline=underline,
        size=size,
    )
    text_element = ET.SubElement(run, qn(W, "t"))
    text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_element.text = text
    return run


def add_paragraph(
    body: ET.Element,
    text: str = "",
    *,
    align: str | None = None,
    bold: bool = False,
    italic: bool = False,
    size: int = 20,
    space_before: int = 0,
    space_after: int = 100,
    line: int = 240,
    keep_next: bool = False,
    first_line: int | None = None,
) -> ET.Element:
    paragraph = ET.SubElement(body, qn(W, "p"))
    properties = ET.SubElement(paragraph, qn(W, "pPr"))
    if align:
        justification = ET.SubElement(properties, qn(W, "jc"))
        justification.set(qn(W, "val"), align)
    spacing = ET.SubElement(properties, qn(W, "spacing"))
    spacing.set(qn(W, "before"), str(space_before))
    spacing.set(qn(W, "after"), str(space_after))
    spacing.set(qn(W, "line"), str(line))
    spacing.set(qn(W, "lineRule"), "auto")
    if keep_next:
        ET.SubElement(properties, qn(W, "keepNext"))
    if first_line is not None:
        indentation = ET.SubElement(properties, qn(W, "ind"))
        indentation.set(qn(W, "firstLine"), str(first_line))
    if text:
        add_run(paragraph, text, bold=bold, italic=italic, size=size)
    return paragraph


def set_cell_text(
    cell: ET.Element,
    text: str,
    *,
    bold: bool = False,
    underline: bool = False,
    align: str = "center",
    size: int = 16,
) -> None:
    paragraph = ET.SubElement(cell, qn(W, "p"))
    properties = ET.SubElement(paragraph, qn(W, "pPr"))
    justification = ET.SubElement(properties, qn(W, "jc"))
    justification.set(qn(W, "val"), align)
    spacing = ET.SubElement(properties, qn(W, "spacing"))
    spacing.set(qn(W, "before"), "0")
    spacing.set(qn(W, "after"), "0")
    spacing.set(qn(W, "line"), "200")
    add_run(paragraph, text, bold=bold, underline=underline, size=size)


def add_cell(
    row: ET.Element,
    text: str,
    *,
    width: int,
    bold: bool = False,
    underline: bool = False,
    align: str = "center",
    grid_span: int | None = None,
    vertical_merge: str | None = None,
    size: int = 16,
) -> ET.Element:
    cell = ET.SubElement(row, qn(W, "tc"))
    properties = ET.SubElement(cell, qn(W, "tcPr"))
    cell_width = ET.SubElement(properties, qn(W, "tcW"))
    cell_width.set(qn(W, "w"), str(width))
    cell_width.set(qn(W, "type"), "dxa")
    if grid_span:
        span = ET.SubElement(properties, qn(W, "gridSpan"))
        span.set(qn(W, "val"), str(grid_span))
    if vertical_merge:
        merge = ET.SubElement(properties, qn(W, "vMerge"))
        if vertical_merge == "restart":
            merge.set(qn(W, "val"), "restart")
    vertical = ET.SubElement(properties, qn(W, "vAlign"))
    vertical.set(qn(W, "val"), "center")
    set_cell_text(
        cell,
        text,
        bold=bold,
        underline=underline,
        align=align,
        size=size,
    )
    return cell


def set_cell_bottom_border(cell: ET.Element, *, size: int = 8) -> None:
    properties = cell.find(qn(W, "tcPr"))
    borders = ET.SubElement(properties, qn(W, "tcBorders"))
    bottom = ET.SubElement(borders, qn(W, "bottom"))
    bottom.set(qn(W, "val"), "single")
    bottom.set(qn(W, "sz"), str(size))
    bottom.set(qn(W, "space"), "0")
    bottom.set(qn(W, "color"), "000000")


def add_table(
    body: ET.Element,
    *,
    grid_widths: list[int],
) -> tuple[ET.Element, list[ET.Element]]:
    table = ET.SubElement(body, qn(W, "tbl"))
    properties = ET.SubElement(table, qn(W, "tblPr"))
    width = ET.SubElement(properties, qn(W, "tblW"))
    width.set(qn(W, "w"), "0")
    width.set(qn(W, "type"), "auto")
    layout = ET.SubElement(properties, qn(W, "tblLayout"))
    layout.set(qn(W, "type"), "fixed")
    alignment = ET.SubElement(properties, qn(W, "jc"))
    alignment.set(qn(W, "val"), "center")
    borders = ET.SubElement(properties, qn(W, "tblBorders"))
    for name, value, size in (
        ("top", "single", 10),
        ("left", "nil", 0),
        ("bottom", "single", 10),
        ("right", "nil", 0),
        ("insideH", "nil", 0),
        ("insideV", "nil", 0),
    ):
        border = ET.SubElement(borders, qn(W, name))
        border.set(qn(W, "val"), value)
        border.set(qn(W, "sz"), str(size))
        border.set(qn(W, "color"), "000000")
    grid = ET.SubElement(table, qn(W, "tblGrid"))
    columns = []
    for grid_width in grid_widths:
        column = ET.SubElement(grid, qn(W, "gridCol"))
        column.set(qn(W, "w"), str(grid_width))
        columns.append(column)
    return table, columns


def add_caption(body: ET.Element, text: str) -> None:
    add_paragraph(
        body,
        text,
        align="left",
        size=16,
        space_before=40,
        space_after=140,
        line=200,
    )


def format_value(value: float) -> str:
    return f"{value:.2f}"


def result_lookup(development: dict) -> dict[tuple[str, str], dict]:
    return {
        (row["model"], row["dataset"]): row
        for row in development["results"]
    }


def build_document(root: Path) -> ET.Element:
    development = load(root / "artifacts/results/DEVELOPMENT_RESULTS.json")
    paper = load(root / "artifacts/results/PAPER_REPORT.json")
    lookup = result_lookup(development)

    document = ET.Element(qn(W, "document"))
    body = ET.SubElement(document, qn(W, "body"))

    add_paragraph(
        body,
        "MVCG-RAG：面向知识冲突消解的多视角交叉佐证图检索增强生成方法",
        align="center",
        bold=True,
        size=30,
        space_after=180,
        line=300,
    )

    add_paragraph(
        body,
        "摘要",
        align="center",
        bold=True,
        size=22,
        keep_next=True,
        space_after=80,
    )
    abstract = (
        "知识冲突消解是提升检索增强生成可靠性的重要环节。传统 RAG 通常由生成模型隐式融合检索知识与参数化知识，"
        "而现有冲突处理方法对多路信息的来源依赖和证据充分性建模不足，容易把同一来源反复产生的信息误认为来自不同视角的相互佐证，因而在冲突或证据不足时产生不可靠回答。针对上述问题，提出了一种面向知识冲突消解的多视角"
        "交叉佐证图检索增强生成方法 MVCG-RAG。该方法受人在冲突信息处理中辨别来源、交叉核验和充分后决策的过程启发。具体来说，首先分别分析检索上下文与模型内部知识，"
        "使二者之间的潜在冲突显式化，并从完整上下文、局部证据和双层语义关联三个层面进行交叉验证。然后，将不同推理与验证结果及其来源组织为交叉佐证图，通过来源"
        "约束限制同一来源的重复贡献，避免信息被反复放大；当上下文结论通过面向当前问题的语义验证后，移除与该结论不一致的基础推理路径。最后，根据交叉佐证是否明确以及证据是否充分，决定输出确定性回答或采用保守结果。该方法无需"
        "任务训练，可在零样本条件下应用于不同基座大模型。在三种基座模型和四个冲突数据集上的实验结果表明，MVCG-RAG 较匹配的已发表方法平均提升 5.86 个百分点，"
        "验证了所提方法的有效性。"
    )
    add_paragraph(
        body,
        abstract,
        align="both",
        size=20,
        first_line=360,
        space_after=180,
    )

    add_paragraph(
        body,
        "实验结果",
        bold=True,
        size=24,
        keep_next=True,
        space_before=120,
        space_after=80,
    )
    add_paragraph(
        body,
        "实验设置",
        bold=True,
        size=21,
        keep_next=True,
        space_before=80,
        space_after=60,
    )
    setup = (
        "我们在四个知识冲突数据集上评估 MVCG-RAG：FaithEval（1,000 个样本）、"
        "RealtimeQA（113 个样本）、MuSiQue-Negative（1,772 个样本）和 "
        "SQuAD-Negative（1,769 个样本）。此外，使用 Llama-3.1-8B-Instruct 在无冲突的 "
        "MuSiQue-Golden 和 SQuAD-Golden 上进行评测。所有实验均采用温度系数 0、12 个"
        "并行 worker，不进行任务训练，不使用示例，也不根据标准答案进行路由或图结构调整。"
        "准确率由 TruthfulRAG 公开的原始评价器计算。为保证基座模型匹配，Llama 的结果与"
        "已发表的 FaithfulRAG 结果对比；Qwen2.5-7B-Instruct 和精确的 "
        "Mistral-7B-Instruct-v0.3 检查点则与已发表的 TruthfulRAG 结果对比。由于对比方法"
        "未公开逐样本预测，我们仅报告点估计差值，不宣称与外部方法之间的配对显著性。"
    )
    add_paragraph(body, setup, align="both", size=20, first_line=360)

    add_paragraph(
        body,
        "总体性能",
        bold=True,
        size=21,
        keep_next=True,
        space_before=100,
        space_after=60,
    )
    add_paragraph(
        body,
        "表 1 给出了知识冲突消解主实验结果。在每个基座模型和每个数据集上，"
        "MVCG-RAG 均超过相应的已发表对比方法。在 Llama 上，本方法相较 "
        "FaithfulRAG 平均提升 5.33 个百分点；在 Qwen 上，相较 TruthfulRAG 平均提升 "
        "6.88 个百分点；在精确的 Mistral v0.3 检查点上，相较 TruthfulRAG 平均提升 "
        "5.38 个百分点。最大的单项提升出现在 Qwen 的 RealtimeQA 上，提升幅度为 "
        "10.62 个百分点；Mistral v0.3 在 RealtimeQA 上也由已发表的 81.40 提升至 88.50。",
        align="both",
        size=20,
        first_line=360,
    )

    main_table, _ = add_table(
        body,
        grid_widths=[1850, 1900, 850, 900, 900, 900, 650, 650],
    )
    header_one = ET.SubElement(main_table, qn(W, "tr"))
    header_cells = [
        add_cell(header_one, "方法", width=1850, bold=True, vertical_merge="restart"),
        add_cell(
            header_one,
            "基座大模型",
            width=1900,
            bold=True,
            vertical_merge="restart",
        ),
        add_cell(
            header_one,
            "数据集",
            width=3550,
            bold=True,
            grid_span=4,
        ),
        add_cell(header_one, "平均", width=650, bold=True, vertical_merge="restart"),
        add_cell(header_one, "提升", width=650, bold=True, vertical_merge="restart"),
    ]
    header_two = ET.SubElement(main_table, qn(W, "tr"))
    add_cell(header_two, "", width=1850, vertical_merge="continue")
    add_cell(header_two, "", width=1900, vertical_merge="continue")
    for label, width in (
        ("FaithEval", 850),
        ("MuSiQue", 900),
        ("RealtimeQA", 900),
        ("SQuAD", 900),
    ):
        add_cell(header_two, label, width=width, bold=True)
    add_cell(header_two, "", width=650, vertical_merge="continue")
    add_cell(header_two, "", width=650, vertical_merge="continue")
    for cell in list(header_two):
        if cell.tag == qn(W, "tc"):
            set_cell_bottom_border(cell)

    comparisons = [
        {
            "baseline_method": "FaithfulRAG（已发表）",
            "ours_method": "MVCG-RAG（本文）",
            "backbone": "Llama-3.1-8B-Instruct",
            "model": "llama",
            "baseline": [79.8, 79.9, 81.4, 86.3],
        },
        {
            "baseline_method": "TruthfulRAG（已发表）",
            "ours_method": "MVCG-RAG（本文）",
            "backbone": "Qwen2.5-7B-Instruct",
            "model": "qwen",
            "baseline": [73.2, 79.1, 82.3, 78.7],
        },
        {
            "baseline_method": "TruthfulRAG（已发表）",
            "ours_method": "MVCG-RAG（本文）",
            "backbone": "Mistral-7B-Instruct-v0.3",
            "model": "mistral",
            "baseline": [81.9, 79.3, 81.4, 82.7],
        },
    ]
    dataset_order = ["faitheval", "musique_negative", "timeqa", "squad_negative"]
    for comparison in comparisons:
        baseline = comparison["baseline"]
        baseline_average = sum(baseline) / len(baseline)
        baseline_row = ET.SubElement(main_table, qn(W, "tr"))
        add_cell(
            baseline_row,
            comparison["baseline_method"],
            width=1850,
            align="left",
            size=15,
        )
        add_cell(
            baseline_row,
            comparison["backbone"],
            width=1900,
            align="left",
            size=15,
        )
        for value, width in zip(baseline, [850, 900, 900, 900]):
            add_cell(
                baseline_row,
                format_value(value),
                width=width,
                underline=True,
                size=15,
            )
        add_cell(
            baseline_row,
            format_value(baseline_average),
            width=650,
            underline=True,
            size=15,
        )
        add_cell(baseline_row, "-", width=650, size=15)

        ours = [
            float(lookup[(comparison["model"], dataset)]["metrics"]["acc"])
            for dataset in dataset_order
        ]
        ours_average = sum(ours) / len(ours)
        ours_row = ET.SubElement(main_table, qn(W, "tr"))
        add_cell(
            ours_row,
            comparison["ours_method"],
            width=1850,
            align="left",
            bold=True,
            size=15,
        )
        add_cell(
            ours_row,
            comparison["backbone"],
            width=1900,
            align="left",
            bold=True,
            size=15,
        )
        for value, width in zip(ours, [850, 900, 900, 900]):
            add_cell(
                ours_row,
                format_value(value),
                width=width,
                bold=True,
                size=15,
            )
        add_cell(
            ours_row,
            format_value(ours_average),
            width=650,
            bold=True,
            size=15,
        )
        add_cell(
            ours_row,
            f"+{ours_average - baseline_average:.2f}",
            width=650,
            bold=True,
            size=15,
        )
        for cell in list(ours_row):
            if cell.tag == qn(W, "tc"):
                set_cell_bottom_border(cell, size=5)

    add_caption(
        body,
        "表 1：MVCG-RAG 与匹配的已发表基线在四个知识冲突数据集上的准确率"
        "（ACC）对比，共使用三种代表性基座大模型。每个基座模型在各数据集上的最优结果"
        "以粗体标注，已发表对比结果以下划线标注。“平均”表示四个数据集的算术平均，“提升”"
        "表示相对于匹配已发表方法的平均提升。MuSiQue 和 SQuAD 对应其 Negative 冲突文件。",
    )

    add_paragraph(
        body,
        "无冲突上下文上的性能",
        bold=True,
        size=21,
        keep_next=True,
        space_before=100,
        space_after=60,
    )
    add_paragraph(
        body,
        "我们进一步检验当检索上下文不存在知识冲突时，所提冲突消解机制能否保持原有回答"
        "能力。遵循 FaithfulRAG 和 TruthfulRAG 的报告方式，Golden 数据集的结果单独列于表 2。"
        "MVCG-RAG 在 MuSiQue-Golden 和 SQuAD-Golden 上分别达到 92.72 和 98.42，较已发表的 "
        "FaithfulRAG 分别提升 7.42 和 1.82 个百分点。与本项目保留的同配置纯上下文 RAG "
        "结果相比，分别提升 1.58 和 0.57 个百分点。",
        align="both",
        size=20,
        first_line=360,
    )

    golden_table, _ = add_table(body, grid_widths=[1900, 1900, 1900, 2200])
    golden_header_one = ET.SubElement(golden_table, qn(W, "tr"))
    add_cell(
        golden_header_one,
        "数据集",
        width=1900,
        bold=True,
        vertical_merge="restart",
    )
    add_cell(
        golden_header_one,
        "方法",
        width=6000,
        bold=True,
        grid_span=3,
    )
    golden_header_two = ET.SubElement(golden_table, qn(W, "tr"))
    add_cell(golden_header_two, "", width=1900, vertical_merge="continue")
    add_cell(
        golden_header_two,
        "同配置纯上下文 RAG",
        width=1900,
        bold=True,
        size=15,
    )
    add_cell(
        golden_header_two,
        "FaithfulRAG（已发表）",
        width=1900,
        bold=True,
        size=15,
    )
    add_cell(
        golden_header_two,
        "MVCG-RAG（本文）",
        width=2200,
        bold=True,
        size=15,
    )
    for cell in list(golden_header_two):
        if cell.tag == qn(W, "tc"):
            set_cell_bottom_border(cell)

    golden_rows = [
        ("MuSiQue-Golden", "musique_golden", 85.3),
        ("SQuAD-Golden", "squad_golden", 96.6),
    ]
    for label, dataset, faithful_value in golden_rows:
        row_data = lookup[("llama", dataset)]
        plain = float(row_data["variants"]["plain_context"]["acc"])
        full = float(row_data["metrics"]["acc"])
        row = ET.SubElement(golden_table, qn(W, "tr"))
        add_cell(row, label, width=1900, align="left", size=16)
        add_cell(row, format_value(plain), width=1900, underline=True, size=16)
        add_cell(row, format_value(faithful_value), width=1900, size=16)
        add_cell(
            row,
            f"{full:.2f} (+{full - plain:.2f} / +{full - faithful_value:.2f})",
            width=2200,
            bold=True,
            size=16,
        )

    add_caption(
        body,
        "表 2：以 Llama-3.1-8B-Instruct 为基座模型时，各方法在无冲突 Golden 上下文上的性能"
        "对比。每个数据集的最优结果以粗体标注，同配置纯上下文 RAG 结果以下划线标注。括号中的"
        "两个数字依次表示相对于同配置纯上下文 RAG 和已发表 FaithfulRAG 结果的变化。",
    )

    add_paragraph(
        body,
        "消融实验",
        bold=True,
        size=21,
        keep_next=True,
        space_before=100,
        space_after=60,
    )
    add_paragraph(
        body,
        "表 3 报告了基于 Llama 的全部冻结回放变体，包括组件消融、容量粒度敏感性和等调用预算投票对照。"
        "所有变体均复用完全相同且已冻结的大模型观测与语义观测，因此对比隔离的是决策机制，而非模型调用预算。"
        "移除查询语义锚点、局部语义锚点和双层语义锚定后，平均准确率分别下降 0.95、0.62 和 1.42 个百分点；"
        "移除查询确认路径封锁后平均下降 0.52 个百分点，其中 FaithEval 下降 1.90 个百分点。"
        "移除保守决策的影响最大，平均下降 3.33 个百分点；仅保留图决策前结果时平均下降 2.09 个百分点。"
        "证据充分性保护使 RealtimeQA 提升 3.54 个百分点。此外，共享基座模型容量在四个冲突数据集上的平均值高出完整方法 0.19 个百分点，"
        "但会使 MuSiQue-Golden 下降 0.96 个百分点，表明过度合并大模型来源存在冲突收益与无冲突保真之间的权衡。",
        align="both",
        size=20,
        first_line=360,
    )

    ablation_table, _ = add_table(
        body,
        grid_widths=[2400, 1100, 1200, 1200, 1200, 850],
    )
    ablation_header = ET.SubElement(ablation_table, qn(W, "tr"))
    for label, width in (
        ("方法", 2400),
        ("FaithEval", 1100),
        ("MuSiQue", 1200),
        ("RealtimeQA", 1200),
        ("SQuAD", 1200),
        ("平均", 850),
    ):
        cell = add_cell(ablation_header, label, width=width, bold=True, size=15)
        set_cell_bottom_border(cell)
    ablations = [
        ("标准 RAG", "plain_context"),
        ("等预算多数投票", "majority_same_budget"),
        ("按来源分组投票", "provenance_grouped_vote"),
        ("仅冲突验证（无图决策）", "scope_only"),
        ("移除查询语义锚点", "without_query_semantic"),
        ("移除局部语义锚点", "without_local_semantic"),
        ("移除双层语义锚定", "without_semantic_bridge"),
        ("移除来源容量约束", "without_provenance_capacity"),
        ("共享基座模型容量", "shared_backbone_capacity"),
        ("扁平化图拓扑", "without_typed_topology"),
        ("移除查询确认路径封锁", "without_directional_cut"),
        ("移除保守决策", "without_risk_projection"),
        ("移除证据充分性保护", "without_idk_protection"),
        ("完整方法", "full"),
    ]
    for label, variant in ablations:
        values = [
            float(lookup[("llama", dataset)]["variants"][variant]["acc"])
            for dataset in dataset_order
        ]
        row = ET.SubElement(ablation_table, qn(W, "tr"))
        is_full = variant == "full"
        add_cell(row, label, width=2400, align="left", bold=is_full, size=15)
        for value, width in zip(values, [1100, 1200, 1200, 1200]):
            add_cell(
                row,
                format_value(value),
                width=width,
                bold=is_full,
                size=15,
            )
        add_cell(
            row,
            format_value(sum(values) / len(values)),
            width=850,
            bold=is_full,
            size=15,
        )

    add_caption(
        body,
        "表 3：以 Llama-3.1-8B-Instruct 为基座模型的 MVCG-RAG 完整消融、容量敏感性与等预算投票对照。"
        "所有结果均以准确率（ACC）表示，并复用相同且已冻结的大模型与语义观测，不改变模型调用预算。"
        "MuSiQue 和 SQuAD 表示其 Negative 冲突文件。",
    )

    add_paragraph(
        body,
        "证据充分性分析",
        bold=True,
        size=21,
        keep_next=True,
        space_before=100,
        space_after=60,
    )
    add_paragraph(
        body,
        "RealtimeQA 包含 63 个可回答样本和 50 个证据不充分样本。如表 4 所示，固定的三来源"
        "证据充分性规则在三种基座模型上均带来提升。该机制对 Mistral v0.3 的改善尤为显著："
        "移除保护机制后，模型容易在证据不充分时生成缺少支持的确定性答案。完整方法在 Mistral "
        "的可回答子集和证据不充分子集上分别达到 93.65 和 82.00 的准确率，而移除该机制时"
        "整体准确率仅为 64.60。",
        align="both",
        size=20,
        first_line=360,
    )

    idk_table, _ = add_table(
        body,
        grid_widths=[1600, 1450, 1150, 950, 1450, 1450],
    )
    idk_header = ET.SubElement(idk_table, qn(W, "tr"))
    for label, width in (
        ("基座模型", 1600),
        ("移除保护", 1450),
        ("完整方法", 1150),
        ("提升", 950),
        ("可回答子集", 1450),
        ("证据不充分子集", 1450),
    ):
        cell = add_cell(idk_header, label, width=width, bold=True, size=15)
        set_cell_bottom_border(cell)
    display_names = {
        "llama": "Llama-3.1-8B",
        "qwen": "Qwen2.5-7B",
        "mistral": "Mistral-7B-v0.3",
    }
    for row_data in paper["idk_sufficiency"]["rows"]:
        row = ET.SubElement(idk_table, qn(W, "tr"))
        add_cell(
            row,
            display_names[row_data["model"]],
            width=1600,
            align="left",
            size=15,
        )
        add_cell(
            row,
            format_value(row_data["without_idk_protection_acc"]),
            width=1450,
            underline=True,
            size=15,
        )
        add_cell(
            row,
            format_value(row_data["full_acc"]),
            width=1150,
            bold=True,
            size=15,
        )
        add_cell(
            row,
            f"+{row_data['effect_pp']:.2f}",
            width=950,
            bold=True,
            size=15,
        )
        add_cell(
            row,
            f"{row_data['answerable_correct']}/{row_data['answerable_n']} "
            f"({row_data['answerable_acc']:.2f})",
            width=1450,
            size=15,
        )
        add_cell(
            row,
            f"{row_data['idk_correct']}/{row_data['idk_n']} "
            f"({row_data['idk_acc']:.2f})",
            width=1450,
            size=15,
        )

    add_caption(
        body,
        "表 4：证据充分性保护机制在 RealtimeQA 上的效果。完整方法以粗体标注，相应消融结果"
        "以下划线标注。“可回答子集”和“证据不充分子集”均报告正确数/总数，括号内为子集准确率。",
    )

    add_paragraph(
        body,
        "综上，实验结果表明，本方法的性能提升并非以牺牲无冲突 Golden 数据集上的性能为代价。"
        "完整系统在所有已配置对比项上均超过匹配的已发表点估计，并在 14 组模型—数据集配置中"
        "有 12 组超过使用相同模型观测的多数投票对比方法。显著性结论仅限于文档内包含的同样本"
        "内部对比；由于 FaithfulRAG 和 TruthfulRAG 未提供逐样本预测，本文不宣称与它们之间"
        "存在配对统计显著性。",
        align="both",
        size=20,
        first_line=360,
        space_before=80,
    )

    section = ET.SubElement(body, qn(W, "sectPr"))
    page_size = ET.SubElement(section, qn(W, "pgSz"))
    page_size.set(qn(W, "w"), "12240")
    page_size.set(qn(W, "h"), "15840")
    margins = ET.SubElement(section, qn(W, "pgMar"))
    for name, value in (
        ("top", "900"),
        ("right", "900"),
        ("bottom", "900"),
        ("left", "900"),
        ("header", "360"),
        ("footer", "360"),
        ("gutter", "0"),
    ):
        margins.set(qn(W, name), value)
    columns = ET.SubElement(section, qn(W, "cols"))
    columns.set(qn(W, "num"), "1")
    return document


def styles_xml() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="SimSun"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
</w:styles>'''


def write_docx(document: ET.Element, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''
    root_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''
    document_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
</Relationships>'''
    settings = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:compat/></w:settings>'''
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{CP}" xmlns:dc="{DC}" xmlns:dcterms="{DCTERMS}" xmlns:xsi="{XSI}">
  <dc:title>MVCG-RAG 摘要与实验结果</dc:title>
  <dc:creator>UGCP Research Project</dc:creator>
  <cp:lastModifiedBy>UGCP Research Project</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>'''
    app = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="{EP}" xmlns:vt="{VT}">
  <Application>Microsoft Office Word</Application>
  <AppVersion>16.0000</AppVersion>
</Properties>'''
    document_xml = ET.tostring(
        document, encoding="utf-8", xml_declaration=True
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml())
        archive.writestr("word/settings.xml", settings)
        archive.writestr("word/_rels/document.xml.rels", document_relationships)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    document = build_document(root)
    write_docx(document, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    run()
