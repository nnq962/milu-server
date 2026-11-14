"""Milu's API: the iPhone app's way to this server's speech to text and local LLM.

Milu keeps its data on the phone. The model never sees it whole: the app sends
the question with the tools it can run (their names, what they do, their
arguments as JSON Schema), the model asks for some of them, the app runs them
on its own data and sends back only their results, and the model answers. So a
new tool is added in the app alone; this server only relays.

    GET  /health          200 once up (no key needed).
    POST /v1/chat         JSON {messages, tools, context} -> {message, ms}.
    POST /v1/transcribe   audio body (M4A/AAC, WebM, WAV), X-Hotwords header
                          (names, URL-encoded, "/" between) -> {text, ms}.

Every /v1 request carries "Authorization: Bearer <key>", one of MILU_API_KEYS.
Nothing said or asked is logged: only the route, the status and the time.
"""

import datetime as dt
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
STT_URL = os.environ.get("STT_URL", "http://speech:8000")
API_KEYS = {key.strip() for key in os.environ.get("MILU_API_KEYS", "").split(",") if key.strip()}
# Per key, per hour: a key copied out of the app cannot run the GPU all day.
CHAT_PER_HOUR = int(os.environ.get("MILU_CHAT_PER_HOUR", "300"))
TRANSCRIBE_PER_HOUR = int(os.environ.get("MILU_TRANSCRIBE_PER_HOUR", "200"))

MAX_CHAT_BYTES = 256_000
MAX_AUDIO_BYTES = 2_000_000
MAX_MESSAGES = 40
MAX_TOOLS = 40
# The model answers in under a second; a long queue behind another user's turn is given up.
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))
STT_TIMEOUT = float(os.environ.get("STT_TIMEOUT", "15"))

WEEKDAYS = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]


# MARK: - Limits


class RateLimit:
    """Requests per key in the last hour, in memory: one process serves the API."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.seen: dict[str, list[float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            recent = [moment for moment in self.seen.get(key, []) if now - moment < 3600]
            if len(recent) >= self.per_hour:
                self.seen[key] = recent
                return False
            recent.append(now)
            self.seen[key] = recent
            return True


chat_limit = RateLimit(CHAT_PER_HOUR)
transcribe_limit = RateLimit(TRANSCRIBE_PER_HOUR)


# MARK: - The model


def date_anchors(today: dt.date) -> str:
    """The stretches people name, worked out here: the model counts days badly."""
    def span(start: dt.date, end: dt.date) -> str:
        return f"{start.isoformat()} → {end.isoformat()}"

    week_start = today - dt.timedelta(days=today.weekday())
    last_week_start = week_start - dt.timedelta(days=7)
    month_start = today.replace(day=1)
    last_month_end = month_start - dt.timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    next_month = (month_start + dt.timedelta(days=32)).replace(day=1)
    lines = [
        f"- Hôm nay: {today.isoformat()} ({WEEKDAYS[today.weekday()]})",
        f"- Hôm qua: {(today - dt.timedelta(days=1)).isoformat()}",
        f"- Tuần này: {span(week_start, week_start + dt.timedelta(days=6))}",
        f"- Tuần trước: {span(last_week_start, last_week_start + dt.timedelta(days=6))}",
        f"- Tháng này: {span(month_start, next_month - dt.timedelta(days=1))}",
        f"- Tháng trước: {span(last_month_start, last_month_end)}",
        f"- Năm nay: {span(today.replace(month=1, day=1), today.replace(month=12, day=31))}",
        f"- 7 ngày qua: {span(today - dt.timedelta(days=6), today)}",
        f"- 30 ngày qua: {span(today - dt.timedelta(days=29), today)}",
    ]
    return "\n".join(lines)


def names(items, limit=80) -> str:
    clean = [str(item).strip()[:60] for item in (items or []) if str(item).strip()]
    return ", ".join(clean[:limit]) or "(chưa có)"


def system_prompt(context: dict) -> str:
    try:
        today = dt.date.fromisoformat(str(context.get("today", ""))[:10])
    except ValueError:
        today = dt.date.today()
    user = str(context.get("userName") or "").strip()[:40]
    return f"""Bạn là Milu, trợ lý tài chính cá nhân trong app Milu. {"Người dùng tên " + user + ". " if user else ""}Luôn trả lời bằng tiếng Việt, ngắn gọn. "Mình" chỉ là Milu; người dùng luôn là "bạn": viết "bạn đã chi", không bao giờ "mình đã chi" (kể cả khi người dùng tự xưng "mình").

Tính cách: Milu là đứa bạn thân Gen Z lầy lội, hay cà khịa kiểu cú Duolingo: thẳng, lém, trêu khi bạn tiêu hoang, tung hô khi bạn tiết kiệm, nhắc khéo mà dai. Dùng từ Gen Z tự nhiên (ủa, xỉu, căng, keo kiệt đỉnh nóc, ví đang khóc, flex, chill, đỉnh của chóp, hông, nha, nè), câu ngắn, có nhịp. Trêu cho vui, không tục, không xúc phạm, không chê ngoại hình hay người thân. Đùa xong vẫn đưa đúng số và một gợi ý có ích. Chuyện nghiêm túc (nợ sắp tới hạn, âm tiền nhiều, người dùng đang buồn) thì bớt đùa, nói rõ ràng, ân cần.
Mỗi câu trả lời dùng 1 đến 2 emoji hợp ngữ cảnh (😳 💸 🫠 😏 🔥 👏 📝 🍜 ☕), chọn emoji phổ biến.

Ví dụ giọng (chỉ để học giọng; KHÔNG chép lại các câu này, mỗi lần nói một kiểu khác; số thật luôn lấy từ công cụ):
- "Ủa **2.300.000đ** cho mua sắm tuần này luôn hả 😳 Shopee chắc thuộc lòng địa chỉ nhà bạn rồi. Thử để giỏ hàng nằm đó 48 tiếng rồi tính tiếp nha."
- Sau khi công cụ ghi xong: "Xong! Ly trà sữa 45.000đ đã vô sổ 📝 Ngon thì ngon chứ cuối tháng đừng than với Milu nha 😏"
- "Tuần này mới chi **320.000đ** thôi á? Keo kiệt đỉnh nóc, Milu tự hào về bạn 👏"

Các mốc thời gian (dùng đúng các ngày này khi gọi công cụ):
{date_anchors(today)}

Danh mục chi: {names(context.get("expenseCategories"))}
Danh mục thu: {names(context.get("incomeCategories"))}
Tài khoản: {names(context.get("accounts"))}

Quy tắc:
- Mọi con số về tiền, giao dịch, số dư, nợ đều phải lấy từ công cụ. Không bao giờ tự đoán hay bịa số.
- Chọn công cụ: tổng chi hay thu một kỳ, chi theo danh mục, trung bình mỗi ngày đã tiêu → summarize_transactions; khoản lớn nhất, đã chi những gì, tìm hay liệt kê giao dịch → list_transactions; số dư, tổng tiền đang có, tài sản ròng → get_accounts; nợ, cho vay, đi vay → get_loans; hạn mức, còn tiêu được bao nhiêu, có vượt không → get_budget; hũ, quỹ, mục tiêu tiết kiệm, tháng này bỏ vào hũ bao nhiêu → get_savings_goals; khoản cố định, hoá đơn định kỳ, tiền nhà, lương hằng tháng → get_recurring; sắp đến hạn, sắp phải trả, quá hạn → get_upcoming; so sánh nhiều tháng, trung bình mỗi tháng, xu hướng, tháng nào chi nhiều nhất, tài sản tăng hay giảm → get_monthly_trend; chuyển tiền giữa các tài khoản, nạp hay rút hũ trong một kỳ, phí chuyển khoản → list_transfers. Câu hỏi cần nhiều thứ thì gọi nhiều công cụ cùng lúc.
- Trường "say" (và "sayAverage") trong kết quả là câu đã tính đúng các số chính: dựa vào đó, không tự cộng trừ lại, và nói rõ khoản nào đã quá hạn.
- Bạn không tự ghi hay xoá được gì: chỉ công cụ làm được. Khi người dùng muốn ghi, thêm hay xoá giao dịch, PHẢI gọi công cụ tương ứng ngay trong lượt này, không hỏi lại để xác nhận (app tự hiện thẻ cho người dùng xem, sửa hoặc huỷ). Chỉ hỏi lại khi thiếu số tiền.
- Không bao giờ nói "đã ghi", "đã xoá", "đã lưu" nếu công cụ chưa trả về kết quả thành công.
- Số tiền là đồng Việt Nam, số nguyên: "60 nghìn", "60k", "60 ca" là 60000; "1 triệu 2", "1 củ 2" là 1200000; "rưỡi" là nửa đơn vị.
- Ghi giao dịch: không rõ chi hay thu thì coi là chi; luôn điền danh mục bằng tên gần nghĩa nhất trong danh sách trên (thưởng → Thưởng, grab → Taxi, xe ôm); tài khoản chỉ điền khi người dùng nói tới.
- Xoá giao dịch: tìm bằng list_transactions, có ref rồi thì gọi delete_transaction ngay, không hỏi lại (app tự hiện nút xác nhận cho người dùng).
- Chữ chỉ trỏ như "ví đó", "tài khoản đó", "khoản đó", "cái vừa rồi" là thứ vừa được nhắc trong hội thoại, không phải tên giống chữ đó trong danh sách.
- Trả lời ngắn: một câu chính, rồi nếu cần thêm tối đa 4 gạch đầu dòng.
- Định dạng (app chỉ hiểu đúng các kiểu này): in đậm bằng **...** cho số tiền hoặc tên quan trọng; mỗi gạch đầu dòng là một dòng bắt đầu bằng "• " (dấu chấm tròn và một dấu cách). KHÔNG dùng "*", "-" hay "+" làm gạch đầu dòng, không gạch đầu dòng lồng nhau, không tiêu đề "#", không bảng, không trích dẫn ">", không code. Không để dòng trống giữa các gạch đầu dòng; một dòng trống giữa câu chính và danh sách là được.
- Viết số tiền kiểu Việt Nam: 1.400.000đ. Viết ngày kiểu 5/11 (thêm năm chỉ khi khác năm nay), hoặc "hôm nay", "hôm qua".
- Câu hỏi về chính app Milu (còn bao nhiêu lượt hỏi, khi nào có lượt mới, cách tính lượt, Milu Pro và giá, quyền riêng tư, dữ liệu lưu ở đâu hay có bị gửi đi không, AI có lưu câu hỏi không, iCloud, sao lưu, đổi máy, xoá dữ liệu, Face ID, điều khoản, tính năng nằm ở đâu, liên hệ, phiên bản): PHẢI gọi get_app_info với topic hợp nhất, rồi trả lời đúng theo kết quả. Không thêm điều kết quả không nói, không đoán số lượt, giá hay ngày. Kết quả không nói tới thì bảo là chưa rõ và chỉ email hỗ trợ (topic support).
- Chuyện quyền riêng tư, dữ liệu, điều khoản: nói nghiêm túc, rõ ràng, trấn an bằng đúng sự thật trong kết quả; đùa nhẹ thôi.
- Câu hỏi không về tiền nong cũng không về app Milu: trả lời ngắn rồi gợi ý quay lại việc tiền nong."""


def to_ollama_message(message: dict) -> dict | None:
    role = message.get("role")
    content = str(message.get("content") or "")[:8000]
    if role == "user":
        return {"role": "user", "content": content}
    if role == "assistant":
        out = {"role": "assistant", "content": content}
        calls = message.get("toolCalls") or []
        if calls:
            out["tool_calls"] = [
                {"function": {"name": str(call.get("name", "")), "arguments": call.get("arguments") or {}}}
                for call in calls[:10]
            ]
        return out
    if role == "tool":
        return {"role": "tool", "tool_name": str(message.get("toolName", "")), "content": content}
    return None


def to_ollama_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": str(tool.get("name", ""))[:64],
            "description": str(tool.get("description", ""))[:1000],
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        },
    }


# A sum of money in a reply: "1.200.000đ", "60k", "2 triệu".
# "đ" alone, not the start of a word: "tháng 9 đi" is no sum.
MONEY = re.compile(r"\d[\d.,]*\s*(?:đ(?!\w)|₫|k\b|nghìn|ngàn|triệu|tr\b|củ\b)", re.IGNORECASE)
BULLET = re.compile(r"^[ \t]*(?:[*+\-•]\s+)+", re.MULTILINE)
# A count of lượt said with no tool behind it ("còn 7 lượt"): only get_app_info knows.
TURNS = re.compile(r"\d+\s*lượt", re.IGNORECASE)
CLAIMS = re.compile(r"đã (?:ghi|xoá|xóa|lưu|thêm)|(?:ghi|xoá|xóa|lưu|thêm) (?:rồi|xong)", re.IGNORECASE)
NUDGE = ("(Nhắc từ hệ thống: bạn chưa gọi công cụ nào. Không được tự đưa ra số tiền hay nói đã ghi/xoá. "
         "Hãy gọi công cụ phù hợp ngay bây giờ, hoặc nói rõ bạn cần thêm thông tin gì.)")


def ask_model(messages: list[dict], tools: list[dict], think: bool, temperature: float = 0) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": think,
        "options": {"temperature": temperature},
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        return json.load(response).get("message") or {}


def chat(body: dict) -> dict:
    messages = [to_ollama_message(m) for m in (body.get("messages") or [])[-MAX_MESSAGES:]]
    messages = [m for m in messages if m]
    if not messages:
        raise ValueError("no messages")
    tools = [to_ollama_tool(t) for t in (body.get("tools") or [])[:MAX_TOOLS] if isinstance(t, dict)]
    convo = [{"role": "system", "content": system_prompt(body.get("context") or {})}, *messages]
    # Since the user's last words, has any tool answered?
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    grounded = any(m["role"] == "tool" for m in messages[last_user + 1:])
    # Choosing a tool, it thinks first (gemma without it made sums up and asked needless
    # questions; with it, every tool right in about a second, 2026-11-12). Writing the answer
    # from a tool's result, it does not: that is quick and plain.
    # The tool chosen at 0, the same every time; the answer written warmer, a new way each time
    # (at 0 it copied the prompt's examples word for word).
    # Telling of the app itself (its lượt, Milu Pro, privacy), the words kept closer to what the
    # app said: warm at 0.8, it added rules of its own (2026-11-15).
    about_app = any(m["role"] == "tool" and m.get("tool_name") == "get_app_info" for m in messages[last_user + 1:])
    temperature = (0.3 if about_app else 0.8) if grounded else 0
    reply = ask_model(convo, tools, think=not grounded, temperature=temperature)
    content = (reply.get("content") or "").strip()
    # A sum, a count of lượt or a "done" with no tool behind it: asked again once, told so.
    if tools and not grounded and not reply.get("tool_calls") and (
            MONEY.search(content) or TURNS.search(content) or CLAIMS.search(content)):
        reply = ask_model([*convo, {"role": "assistant", "content": content}, {"role": "user", "content": NUDGE}],
                          tools, think=True)
    calls = []
    for index, call in enumerate(reply.get("tool_calls") or []):
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append({"id": call.get("id") or f"call_{index}", "name": function.get("name", ""), "arguments": arguments})
    # Emoji are kept (user choice 2026-11-12); the app leaves out any the iPhone cannot draw.
    content = re.sub(r"[ \t]+\n", "\n", re.sub(r"(?<=\S) {2,}", " ", reply.get("content") or "")).strip()
    # List items in the one way the app shows them ("• "), however the model wrote them
    # (it kept writing "* " on warm answers, eval 2026-11-12).
    content = BULLET.sub("• ", content)
    return {"content": content, "toolCalls": calls}


# MARK: - Speech


def transcribe(audio: bytes, hotwords_header: str) -> str:
    request = urllib.request.Request(
        f"{STT_URL}/transcribe",
        data=audio,
        headers={"Content-Type": "application/octet-stream", "X-Hotwords": hotwords_header},
    )
    with urllib.request.urlopen(request, timeout=STT_TIMEOUT) as response:
        raw = (json.load(response).get("text") or "").strip()
    names_given = [urllib.parse.unquote(name).strip() for name in hotwords_header.split("/") if name.strip()]
    return as_sentence(raw, names_given)


SPELLED = re.compile(r"(?<![^\W\d_])[A-Z](?: [A-Z])+(?![^\W\d_])")


def as_sentence(text: str, names_given: list[str]) -> str:
    """The recognizer's capitals as a sentence: lower case, letters said one by one joined
    ("B I D V" → "BIDV"), each name given written as it is spelled, the first letter up."""
    joined = SPELLED.sub(lambda m: m.group(0).replace(" ", ""), text)
    sentence = joined.lower()
    for name in sorted(set(names_given), key=len, reverse=True):
        # Any letters of it said apart: "mo mo" is MoMo, "viet com bank" Vietcombank.
        pattern = r"\s*".join(re.escape(char) for char in name.lower() if not char.isspace())
        if pattern:
            sentence = re.sub(rf"(?<!\w){pattern}(?!\w)", name, sentence)
    return sentence[:1].upper() + sentence[1:]


# MARK: - Server


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _key(self) -> str | None:
        header = self.headers.get("Authorization", "")
        key = header[7:].strip() if header.startswith("Bearer ") else ""
        return key if key and key in API_KEYS else None

    def _read_body(self) -> bytes | None:
        """The whole body, read before any answer: one left unread on a kept-alive connection
        was taken for the start of the next request ("{}POST", 501). Too large, the connection
        is closed instead."""
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_AUDIO_BYTES:
            self.close_connection = True
            return None
        return self.rfile.read(length) if length > 0 else b""

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send(200, {"ok": True, "model": OLLAMA_MODEL})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        started = time.perf_counter()
        status = 500
        raw = self._read_body()
        try:
            if self.path not in ("/v1/chat", "/v1/transcribe"):
                status = 404
                return self._send(404, {"error": "not found"})
            key = self._key()
            if not key:
                status = 401
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/v1/chat":
                if not chat_limit.allow(key):
                    status = 429
                    return self._send(429, {"error": "too many requests"})
                if not raw or len(raw) > MAX_CHAT_BYTES:
                    status = 413
                    return self._send(413, {"error": "body missing or too large"})
                try:
                    body = json.loads(raw)
                    message = chat(body)
                except (json.JSONDecodeError, ValueError):
                    status = 400
                    return self._send(400, {"error": "bad request"})
                status = 200
                return self._send(200, {"message": message, "ms": round((time.perf_counter() - started) * 1000)})
            if not transcribe_limit.allow(key):
                status = 429
                return self._send(429, {"error": "too many requests"})
            audio = raw
            if not audio:
                status = 413
                return self._send(413, {"error": "audio missing or too large"})
            text = transcribe(audio, self.headers.get("X-Hotwords", ""))
            status = 200
            self._send(200, {"text": text, "ms": round((time.perf_counter() - started) * 1000)})
        except urllib.error.HTTPError as error:
            status = 502
            self._send(502, {"error": f"upstream {error.code}"})
        except (urllib.error.URLError, TimeoutError, OSError):
            status = 503
            self._send(503, {"error": "upstream unavailable"})
        finally:
            # What was said or asked is never logged: only the route, the status and the time.
            print(f"{self.path} {status} {round((time.perf_counter() - started) * 1000)}ms", flush=True)

    def log_message(self, format: str, *args) -> None:
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 64
    daemon_threads = True


if __name__ == "__main__":
    if not API_KEYS:
        raise SystemExit("Set MILU_API_KEYS")
    Server(("0.0.0.0", int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
