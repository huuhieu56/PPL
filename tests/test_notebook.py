import json
import re
import shlex
from pathlib import Path

import pytest

from src.cli import build_parser
from src.eval.report import FIGURES, TABLES

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "b_o_c_o_nh_m_3.md"


def test_notebook_cli_commands_parse():
    notebook = json.loads((ROOT / "notebooks" / "colab_pipeline.ipynb").read_text(encoding="utf-8"))
    commands = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for line in "".join(cell["source"]).splitlines():
            if "python -m src.cli" in line:
                commands.append(shlex.split(line.split("python -m src.cli", 1)[1].split("#")[0]))
    assert len(commands) >= 12
    parser = build_parser()
    for arguments in commands:
        parser.parse_args(arguments)


@pytest.mark.skipif(not REPORT.exists(), reason="report file is not in this checkout")
def test_report_chapter_three_references_every_table_and_figure():
    text = REPORT.read_text(encoding="utf-8")
    assert "# CHƯƠNG 3" in text
    for key, title in TABLES.items():
        assert f"Bảng {key}. {title}" in text
    for key, title in FIGURES.items():
        assert f"Hình {key}. {title}" in text
    assert text.index("# CHƯƠNG 3") < text.index("# DANH MỤC TÀI LIỆU THAM KHẢO")
    numbers = re.findall(r"\[\[ĐIỀN:", text)
    assert len(numbers) >= 20
