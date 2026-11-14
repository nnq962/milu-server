"""Chấm điểm Milu: chạy các câu hỏi mẫu (cases.json) qua /v1/chat và so với đáp án.

    MILU_API_URL=https://milu-api.nnqlab.dev MILU_API_KEY=... python3 eval/run.py [--runs 3] [--only r01,w02]

Đọc app.json (danh sách tool và ngữ cảnh đúng như app gửi, xuất từ test của app:
TEST_RUNNER_MILU_EXPORT=<path> xcodebuild test ... -only-testing:MiluTests/AssistantToolsTests).
Ghi kết quả vào eval/results.md. Chỉ dùng thư viện chuẩn.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
# "đ" alone, not the start of a word: "tháng 9 đi" is no sum.
MONEY = re.compile(r"\d[\d.,]*\s*(?:đ(?!\w)|₫|k\b|nghìn|ngàn|triệu|tr\b|củ\b)", re.IGNORECASE)
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿]")


def key(text) -> str:
    """Lower case, no accents: how names are compared."""
    text = unicodedata.normalize("NFD", str(text).lower().replace("đ", "d"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").strip()


def call(url: str, api_key: str, body: dict) -> tuple[dict, int]:
    request = urllib.request.Request(
        f"{url}/v1/chat",
        data=json.dumps(body).encode(),
        # Cloudflare turns away Python's own user agent (403).
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "milu-eval/1"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        reply = json.load(response)["message"]
    return reply, round((time.perf_counter() - started) * 1000)


# MARK: - Matching


def matches(expected, actual) -> bool:
    if isinstance(expected, dict):
        if "any" in expected:
            return any(matches(option, actual) for option in expected["any"])
        if "names" in expected:
            values = actual if isinstance(actual, list) else [actual]
            wanted = [key(name) for name in expected["names"]]
            return any(
                key(value) == w or (key(value) and (key(value) in w or w in key(value)))
                for value in values if value is not None for w in wanted
            )
        if expected.get("absent"):
            return actual is None
    if expected is None:
        return actual in (None, "", [])
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(str(actual).replace(".", "").replace(",", "")) == float(expected)
        except ValueError:
            return False
    return key(actual) == key(expected)


def check_tool_case(case: dict, reply: dict) -> list[str]:
    expect = case["expect"]
    calls = reply.get("toolCalls") or []
    content = reply.get("content") or ""
    want_tool = expect["tool"]
    if not calls:
        if matches(want_tool, "none"):
            return ["tự đưa số tiền khi chưa gọi tool"] if MONEY.search(content) else []
        return [f"không gọi tool (cần {want_tool}); trả lời: {content[:80]!r}"]
    call = calls[0]
    if not matches(want_tool, call["name"]):
        return [f"gọi {call['name']} (cần {want_tool})"]
    problems = []
    arguments = call.get("arguments") or {}
    for name, wanted in (expect.get("args") or {}).items():
        if not matches(wanted, arguments.get(name)):
            problems.append(f"{name}={arguments.get(name)!r} (cần {wanted})")
    return problems


def check_answer_case(case: dict, reply: dict) -> tuple[list[str], list[str]]:
    """Problems that fail the case, and notes that do not."""
    content = (reply.get("content") or "").strip()
    if reply.get("toolCalls"):
        return [f"gọi thêm tool {reply['toolCalls'][0]['name']} thay vì trả lời"], []
    if not content:
        return ["trả lời rỗng"], []
    problems, notes = [], []
    # Words compared without accents or bold: "xóa" is "xoá", "**3** lượt" is "3 lượt".
    lowered = key(content.replace("**", ""))
    answer = case["answer"]
    for text in answer.get("has", []):
        if key(text) not in lowered:
            problems.append(f"thiếu {text!r}")
    for group in answer.get("hasAny", []):
        if not any(key(text) in lowered for text in group):
            problems.append(f"thiếu một trong {group!r}")
    for text in answer.get("not", []):
        if key(text) in lowered:
            problems.append(f"có {text!r}")
    if "noMoneyExcept" in answer:
        allowed = {re.sub(r"\D", "", a) for a in answer["noMoneyExcept"]}
        for found in MONEY.findall(content):
            if re.sub(r"\D", "", found) not in allowed:
                problems.append(f"bịa số {found!r}")
    for line in content.splitlines():
        stripped = line.strip()
        if re.match(r"^[*\-+]\s", stripped):
            problems.append(f"gạch đầu dòng sai kiểu: {stripped[:30]!r}")
        if stripped.startswith("#"):
            problems.append("có tiêu đề #")
    if re.search(r"\bmình (đã )?(chi|tiêu|nhận)", content.lower()):
        problems.append("xưng 'mình' cho người dùng")
    if not EMOJI.search(content):
        notes.append("không có emoji")
    if len(content.splitlines()) > 8:
        notes.append(f"dài ({len(content.splitlines())} dòng)")
    return problems, notes


# MARK: - Running


def messages_for(case: dict, app: dict) -> list[dict]:
    if "toolCall" in case:
        result = case["toolResult"]
        # The app's own answer to get_app_info, as exported with the tools (app.json "appInfo").
        if isinstance(result, dict) and "$appInfo" in result:
            result = app["appInfo"][result["$appInfo"]]
        return [
            {"role": "user", "content": case["q"]},
            {"role": "assistant", "content": "", "toolCalls": [{"id": "c1", **case["toolCall"]}]},
            {"role": "tool", "toolName": case["toolCall"]["name"], "content": json.dumps(result, ensure_ascii=False)},
        ]
    history = list(case.get("history") or [])
    return history if case.get("continue") else [*history, {"role": "user", "content": case["q"]}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    url = os.environ.get("MILU_API_URL", "https://milu-api.nnqlab.dev").rstrip("/")
    api_key = os.environ.get("MILU_API_KEY") or sys.exit("Set MILU_API_KEY")
    app = json.loads((HERE / "app.json").read_text())
    cases = json.loads((HERE / "cases.json").read_text())["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    rows = []
    for case in cases:
        body = {"messages": messages_for(case, app), "tools": app["tools"], "context": app["context"]}
        passes, times, problems, notes, sample = 0, [], [], [], ""
        for _ in range(args.runs):
            try:
                reply, ms = call(url, api_key, body)
            except (urllib.error.URLError, TimeoutError) as error:
                problems.append(f"lỗi gọi API: {error}")
                continue
            times.append(ms)
            if "answer" in case:
                found, run_notes = check_answer_case(case, reply)
                notes.extend(run_notes)
                sample = sample or (reply.get("content") or "")
            else:
                found = check_tool_case(case, reply)
                calls = reply.get("toolCalls") or []
                sample = sample or (json.dumps(calls[0], ensure_ascii=False) if calls else (reply.get("content") or ""))
            if found:
                problems.extend(p for p in found if p not in problems)
            else:
                passes += 1
        rows.append({"case": case, "passes": passes, "ms": statistics.median(times) if times else 0,
                     "problems": problems, "notes": sorted(set(notes)), "sample": sample})
        mark = "✅" if passes == args.runs else ("⚠️" if passes else "❌")
        print(f"{mark} {case['id']:4} {passes}/{args.runs} {rows[-1]['ms']:>5}ms  {case['q'][:48]}"
              + (f"  ← {problems[0]}" if problems else ""), flush=True)

    write_report(rows, args.runs)


def write_report(rows: list[dict], runs: int) -> None:
    total = len(rows) * runs
    passed = sum(r["passes"] for r in rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["case"]["group"], []).append(row)
    lines = [
        f"# Kết quả chấm điểm Milu ({time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"**{passed}/{total} lượt đạt ({passed / total:.0%})**, {len(rows)} câu × {runs} lượt. "
        f"Thời gian trung vị một lượt gọi: {statistics.median([r['ms'] for r in rows]):.0f} ms.",
        "",
        "| Nhóm | Đạt | Tỉ lệ |",
        "|---|---|---|",
    ]
    for name, items in groups.items():
        ok = sum(r["passes"] for r in items)
        lines.append(f"| {name} | {ok}/{len(items) * runs} | {ok / (len(items) * runs):.0%} |")
    lines += ["", "## Câu chưa đạt", ""]
    for row in rows:
        if row["passes"] < runs:
            lines.append(f"- **{row['case']['id']}** `{row['case']['q']}` — {row['passes']}/{runs}: "
                         + "; ".join(row["problems"][:3]) + f"<br>Mẫu: `{row['sample'][:160]}`")
    notes = [r for r in rows if r["notes"]]
    if notes:
        lines += ["", "## Ghi chú (không trừ điểm)", ""]
        lines += [f"- **{r['case']['id']}**: {', '.join(r['notes'])}" for r in notes]
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\n{passed}/{total} ({passed / total:.0%}) — eval/results.md")


if __name__ == "__main__":
    main()
