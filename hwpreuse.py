import re
import json
import logging
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

from flask import Flask, request, send_file, render_template_string
from openai import OpenAI
from lxml import etree

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("hwpx")

client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.openai.com/v1/chat/completions"
)
MODEL = "gpt-5.4"

HP_NS = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS_NS = "http://www.hancom.co.kr/hwpml/2011/section"

TAG_T       = f"{{{HP_NS}}}t"
TAG_P       = f"{{{HP_NS}}}p"
TAG_RUN     = f"{{{HP_NS}}}run"
TAG_TBL     = f"{{{HP_NS}}}tbl"
TAG_TR      = f"{{{HP_NS}}}tr"
TAG_TC      = f"{{{HP_NS}}}tc"
TAG_SUBLIST = f"{{{HP_NS}}}subList"
TAG_SEC     = f"{{{HS_NS}}}sec"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HWPX 압축 해제 / 재패킹
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_hwpx(file_bytes: bytes) -> tuple[Path, etree._Element]:
    tmpdir = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(BytesIO(file_bytes), "r") as zf:
        zf.extractall(tmpdir)
    section_path = tmpdir / "Contents" / "section0.xml"
    if not section_path.exists():
        raise FileNotFoundError("section0.xml이 HWPX 안에 없습니다.")
    return tmpdir, etree.parse(str(section_path)).getroot()


def save_and_pack(tmpdir: Path, root: etree._Element) -> bytes:
    etree.indent(root, space="  ")
    etree.ElementTree(root).write(
        str(tmpdir / "Contents" / "section0.xml"),
        pretty_print=True, xml_declaration=True, encoding="UTF-8",
    )
    buf = BytesIO()
    mimetype_file = tmpdir / "mimetype"
    all_files = sorted(
        p.relative_to(tmpdir).as_posix()
        for p in tmpdir.rglob("*") if p.is_file()
    )
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(mimetype_file, "mimetype", compress_type=zipfile.ZIP_STORED)
        for rel in all_files:
            if rel != "mimetype":
                zf.write(tmpdir / rel, rel, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 템플릿 구조 파악 (위치 맵 생성)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _p_text(p_el: etree._Element) -> str:
    """표를 제외한 문단의 텍스트를 반환."""
    parts = []
    for run in p_el.findall(TAG_RUN):
        if run.find(f".//{TAG_TBL}") is not None:
            continue
        for t in run.findall(TAG_T):
            if t.text:
                parts.append(t.text)
    return "".join(parts)


def _cell_text(tc_el: etree._Element) -> str:
    return "".join(t.text for t in tc_el.iter(TAG_T) if t.text)


def build_doc_map(root: etree._Element) -> tuple[dict, dict, dict]:
    """
    섹션의 직접 자식 <hp:p> 를 순서대로 순회해
    - para_map  : {para_idx: p_element}   (표 아닌 문단)
    - tbl_map   : {tbl_idx:  tbl_element} (표)
    - structure : AI에 전달할 문서 구조 dict

    structure 형태:
      {
        "paragraphs": [{"index": 0, "text": "..."}, ...],
        "tables":     [{"index": 0, "rows": [["셀", ...], ...]}, ...]
      }
    """
    sec = root.find(f".//{TAG_SEC}")
    if sec is None:
        sec = root

    para_map: dict[int, etree._Element] = {}
    tbl_map:  dict[int, etree._Element] = {}
    s_paras = []
    s_tables = []
    para_idx = tbl_idx = 0

    for p in sec:
        if p.tag != TAG_P:
            continue
        tbl_el = p.find(f".//{TAG_TBL}")
        if tbl_el is not None:
            rows = []
            for tr in tbl_el.findall(f".//{TAG_TR}"):
                rows.append([_cell_text(tc) for tc in tr.findall(TAG_TC)])
            tbl_map[tbl_idx] = tbl_el
            s_tables.append({"index": tbl_idx, "rows": rows})
            tbl_idx += 1
        else:
            text = _p_text(p)
            para_map[para_idx] = p
            s_paras.append({"index": para_idx, "text": text})
            para_idx += 1

    return para_map, tbl_map, {"paragraphs": s_paras, "tables": s_tables}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AI 호출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_system_prompt(structure: dict) -> str:
    lines = [
        "당신은 공공기관 행정문서 작성 전문가입니다.",
        "아래 [현재 문서 구조]를 보고 사용자 요청에 맞게 내용을 작성하세요.",
        "변경이 필요한 항목만 결과 JSON에 포함하세요.",
        "설명·마크다운·코드블록 없이 **순수 JSON만** 출력하세요.",
        "",
        "=== 현재 문서 구조 ===",
    ]

    if structure["paragraphs"]:
        lines.append("[문단 목록]")
        for p in structure["paragraphs"]:
            text = p["text"] if p["text"] else "(빈 칸)"
            lines.append(f'  [{p["index"]}] {text}')
        lines.append("")

    if structure["tables"]:
        lines.append("[표 목록]")
        for t in structure["tables"]:
            lines.append(f'  표 {t["index"]}:')
            for r_idx, row in enumerate(t["rows"]):
                cells = " | ".join(f'"{c}"' if c else '(빈칸)' for c in row)
                lines.append(f'    행{r_idx}: {cells}')
        lines.append("")

    lines += [
        "=== 출력 형식 ===",
        '{',
        '  "paragraphs": [',
        '    {"index": <문단번호>, "text": "<새 내용>"},',
        '    ...',
        '  ],',
        '  "tables": [',
        '    {',
        '      "index": <표번호>,',
        '      "start_row": <데이터 시작 행번호 (헤더 다음 행)>,',
        '      "rows": [["값1", "값2", ...], ...]',
        '    }',
        '  ]',
        '}',
        "",
        "주의:",
        "- 변경하지 않을 문단/셀은 포함하지 마세요.",
        "- text 값에 줄바꿈(\\n) 사용 금지.",
        "- 헤더 행(행0)은 rows에 포함하지 마세요.",
    ]
    return "\n".join(lines)


def call_ai(user_prompt: str, structure: dict) -> dict:
    system = build_system_prompt(structure)
    log.info("System:\n%s", system)
    log.info("User: %s", user_prompt)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    text = resp.choices[0].message.content.strip()
    log.info("AI: %s", text[:800])

    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    return json.loads(text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XML 수정: 위치 기반으로 텍스트 교체
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_paragraph(p_el: etree._Element, new_text: str) -> None:
    """
    문단의 텍스트를 교체한다.
    첫 번째 <hp:t>에 전체 텍스트를 넣고 나머지 <hp:t>는 비운다.
    (글자 스타일은 원본 charPrIDRef를 유지)
    """
    t_elements = [t for t in p_el.iter(TAG_T)
                  if t.getparent() is not None
                  and t.getparent().tag == TAG_RUN]

    if not t_elements:
        # <hp:t>가 없으면 첫 run 안에 생성
        run = p_el.find(TAG_RUN)
        if run is None:
            run = etree.SubElement(p_el, TAG_RUN)
            run.set("charPrIDRef", "0")
        t_el = etree.SubElement(run, TAG_T)
        t_el.text = new_text
        return

    # 첫 번째에 새 텍스트, 나머지는 비움
    t_elements[0].text = new_text
    for t_el in t_elements[1:]:
        t_el.text = ""


def apply_table(tbl_el: etree._Element, start_row: int, rows: list[list]) -> None:
    """표의 지정 행부터 데이터를 채운다."""
    tr_list = tbl_el.findall(f".//{TAG_TR}")
    for data_idx, row_data in enumerate(rows):
        tr_idx = start_row + data_idx
        if tr_idx >= len(tr_list):
            break
        tc_list = tr_list[tr_idx].findall(TAG_TC)
        for col_idx, val in enumerate(row_data):
            if col_idx >= len(tc_list):
                break
            _set_cell_text(tc_list[col_idx], str(val))


def _set_cell_text(tc_el: etree._Element, text: str) -> None:
    t_list = list(tc_el.iter(TAG_T))
    if t_list:
        t_list[0].text = text
        for t in t_list[1:]:
            t.text = ""
        return
    # 빈 셀: 구조 생성
    sublist = tc_el.find(f".//{TAG_SUBLIST}")
    if sublist is None:
        return
    p_el = sublist.find(TAG_P)
    if p_el is None:
        return
    run_el = etree.SubElement(p_el, TAG_RUN)
    run_el.set("charPrIDRef", "0")
    t_el = etree.SubElement(run_el, TAG_T)
    t_el.text = text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 처리 흐름
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process(file_bytes: bytes, user_prompt: str) -> bytes:
    tmpdir = None
    try:
        # 1. HWPX 압축 해제 + XML 파싱
        tmpdir, root = extract_hwpx(file_bytes)

        # 2. 문서 구조 파악 (문단 위치 맵 + 표 위치 맵 + AI 전달용 구조)
        para_map, tbl_map, structure = build_doc_map(root)
        log.info("문단 %d개, 표 %d개 인식", len(para_map), len(tbl_map))

        # 3. AI 호출: 현재 구조 + 사용자 요청 → 변경 내용 JSON 반환
        data = call_ai(user_prompt, structure)

        # 4. 문단 업데이트
        for item in data.get("paragraphs", []):
            idx = item.get("index")
            text = item.get("text", "")
            if idx in para_map:
                apply_paragraph(para_map[idx], text)
                log.info("문단[%d] → %s", idx, text[:60])
            else:
                log.warning("문단 인덱스 %d 없음", idx)

        # 5. 표 업데이트
        for tbl_data in data.get("tables", []):
            idx = tbl_data.get("index")
            start_row = tbl_data.get("start_row", 1)
            rows = tbl_data.get("rows", [])
            if idx in tbl_map:
                apply_table(tbl_map[idx], start_row, rows)
                log.info("표[%d] %d행 업데이트", idx, len(rows))
            else:
                log.warning("표 인덱스 %d 없음", idx)

        # 6. 저장 + HWPX 재패킹
        return save_and_pack(tmpdir, root)

    finally:
        if tmpdir and tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 웹 UI 및 라우트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HTML = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>AI HWPX 문서 생성기</title>
<style>
body{font-family:'Malgun Gothic',sans-serif;max-width:680px;margin:40px auto;padding:0 20px;background:#f8f9fa}
h2{color:#1a3a5c;margin-bottom:4px}
.sub{color:#666;font-size:13px;margin-bottom:20px}
label{font-weight:bold;display:block;margin-bottom:6px}
textarea{width:100%;height:130px;font-size:14px;padding:8px;border:1px solid #ccc;border-radius:4px;box-sizing:border-box}
button{background:#1a3a5c;color:#fff;padding:10px 28px;border:none;border-radius:4px;font-size:15px;cursor:pointer;margin-top:12px}
button:hover{background:#2a5a8c}
.hint{color:#888;font-size:12px;margin-top:5px}
.err{color:#c00;font-weight:bold;margin:12px 0;padding:10px;background:#fff0f0;border-radius:4px}
.card{background:#fff;border-radius:8px;padding:24px;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:16px}
</style></head><body>
<h2>AI HWPX 문서 생성기</h2>
<p class="sub">HWPX 양식을 올리고 원하는 내용을 입력하면, AI가 양식 구조를 파악하여 문서를 완성합니다.</p>
{% if error %}<p class="err">오류: {{ error }}</p>{% endif %}
<div class="card">
<form method="post" enctype="multipart/form-data">
  <label>HWPX 양식 파일</label>
  <input type="file" name="file" accept=".hwpx" required><br><br>
  <label>작성 요청</label>
  <textarea name="prompt" placeholder="예: 2026년 2차 윤리위원회 운영결과 보고서를 작성해줘. 회의일시는 2026년 4월 10일이고 참석자는 5명이야."></textarea>
  <p class="hint">양식에 별도 표시 없이도 AI가 구조를 읽어 내용을 채웁니다.</p>
  <button type="submit">문서 생성</button>
</form>
</div>
</body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(HTML, error=None)

    f = request.files.get("file")
    prompt = request.form.get("prompt", "").strip()

    if not f or not f.filename:
        return render_template_string(HTML, error="HWPX 파일을 업로드해주세요.")
    if not prompt:
        return render_template_string(HTML, error="작성 요청 내용을 입력해주세요.")

    try:
        result = process(f.read(), prompt)
        return send_file(
            BytesIO(result),
            as_attachment=True,
            download_name="완성_문서.hwpx",
            mimetype="application/octet-stream",
        )
    except json.JSONDecodeError as e:
        return render_template_string(HTML, error=f"AI 응답 형식 오류: {e}")
    except Exception as e:
        log.exception("처리 오류")
        return render_template_string(HTML, error=str(e))


@app.route("/api/generate", methods=["POST"])
def api_generate():
    f = request.files.get("file")
    prompt = request.form.get("prompt", "")
    if not f:
        return {"error": "file required"}, 400
    if not prompt:
        return {"error": "prompt required"}, 400
    try:
        result = process(f.read(), prompt)
        return send_file(BytesIO(result), as_attachment=True, download_name="result.hwpx")
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  AI HWPX 문서 생성기")
    print("  http://localhost:8008")
    print(f"  모델: {MODEL}")
    print("=" * 50 + "\n")
    app.run(port=8008, debug=True, use_reloader=False)
