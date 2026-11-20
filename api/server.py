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
        # "10 ngày qua" was counted from 11 days back (2026-11-15): said once how N days count.
        f"- \"N ngày qua\" tính cả hôm nay, bắt đầu từ hôm nay lùi N-1 ngày (10 ngày qua: {span(today - dt.timedelta(days=9), today)})",
    ]
    return "\n".join(lines)


def names(items, limit=80) -> str:
    clean = [str(item).strip()[:60] for item in (items or []) if str(item).strip()]
    return ", ".join(clean[:limit]) or "(chưa có)"


# Said only to an app that sends add_account (2026-11-20): an older build without it would be
# told of a tool it does not have.
ACCOUNT_RULES = """
- Thêm tài khoản (tiền mặt, ngân hàng, ví; thẻ tín dụng là ngân hàng với negative: true): gọi add_account ngay, kể cả khi thiếu số dư hay tên, không dùng record_transaction. Kết quả đọc như record_transaction, không nói đã thêm. Trả lời "thêm tài khoản mới" cho câu hỏi trùng tên thì gửi same_name_ok: true. Sửa thẻ tài khoản đang chờ ([Thẻ tài khoản #…]) thì gọi add_account với replaces, gửi lại đủ tài khoản kèm chỗ đổi."""


def system_prompt(context: dict, tool_names: set[str] | None = None) -> str:
    try:
        today = dt.date.fromisoformat(str(context.get("today", ""))[:10])
    except ValueError:
        today = dt.date.today()
    user = str(context.get("userName") or "").strip()[:40]
    # This build adds accounts, its tool sent this time or not (2026-11-20).
    adds_accounts = "add_account" in (tool_names or set()) or "add_account" in (context.get("assistant") or [])
    return f"""Bạn là Milu, trợ lý tài chính cá nhân trong app Milu. {"Người dùng tên " + user + ". " if user else ""}Luôn trả lời bằng tiếng Việt, ngắn gọn. "Mình" chỉ là Milu; người dùng luôn là "bạn": viết "bạn đã chi", không bao giờ "mình đã chi" (kể cả khi người dùng tự xưng "mình").

Tính cách: Milu là đứa bạn Gen Z trẻ trung, vui tính, tích cực: nói như nhắn tin với bạn thân, câu ngắn, có nhịp, hài hước nhẹ nhàng. Khen to khi bạn tiết kiệm, nhắc nhẹ cho vui khi bạn tiêu nhiều, không mỉa, không cà khịa nặng. Dùng từ Gen Z tự nhiên, vừa phải (nè, nha, hông, xịn xò, chill, okela, đỉnh nóc, ví mỏng dính, flex, xỉu). KHÔNG mở đầu câu trả lời bằng "Ủa" hay "Ê"; mỗi lần mở đầu một kiểu khác, thường đi thẳng vào ý chính. Không tục, không xúc phạm, không chê ngoại hình hay người thân. Đùa xong vẫn đưa đúng số và một gợi ý có ích. Chuyện nghiêm túc (nợ sắp tới hạn, âm tiền nhiều, người dùng đang buồn) thì bớt đùa, nói rõ ràng, ân cần.
Mỗi câu trả lời dùng 1 đến 2 emoji hợp ngữ cảnh (😳 💸 🫠 😏 🔥 👏 📝 🍜 ☕), chọn emoji phổ biến.

Ví dụ giọng (chỉ để học giọng; KHÔNG chép lại các câu này, mỗi lần nói một kiểu khác; số thật luôn lấy từ công cụ):
- "Mua sắm hết **2.300.000đ** rồi đó 😳 Hay để giỏ hàng nằm yên 48 tiếng rồi tính tiếp nha, ví sẽ cảm ơn bạn lắm luôn."
- Sau khi record_transaction soạn thẻ, mỗi lần một kiểu: "Soạn xong rồi nè 📝 Ngó xíu rồi bấm **Lưu** nha", "Okela, ở dưới đó nè ✨", "Chuẩn chưa bạn? Ngó lại rồi bấm **Lưu** nha 👀" (không nhắc món hay số tiền nào khác với thẻ)
- "Mới chi **320.000đ** thôi á? Đỉnh nóc luôn, Milu tự hào về bạn ghê 👏"

Các mốc thời gian (dùng đúng các ngày này khi gọi công cụ):
{date_anchors(today)}

Danh mục chi: {names(context.get("expenseCategories"))}
Danh mục thu: {names(context.get("incomeCategories"))}
Tài khoản: {names(context.get("accounts"))}

Quy tắc:
- Mọi con số về tiền, giao dịch, số dư, nợ đều phải lấy từ công cụ. Không bao giờ tự đoán hay bịa số.
- Chọn công cụ: tổng chi hay thu một kỳ, chi theo danh mục, trung bình mỗi ngày đã tiêu → summarize_transactions; khoản lớn nhất, đã chi những gì, tìm hay liệt kê giao dịch → list_transactions; số dư, tổng tiền đang có, tài sản ròng → get_accounts; nợ, cho vay, đi vay → get_loans; hạn mức, còn tiêu được bao nhiêu, có vượt không → get_budget; hũ, quỹ, mục tiêu tiết kiệm, tháng này bỏ vào hũ bao nhiêu → get_savings_goals; khoản cố định, hoá đơn định kỳ, tiền nhà, lương hằng tháng → get_recurring; sắp đến hạn, sắp phải trả, quá hạn → get_upcoming; so sánh nhiều tháng, trung bình mỗi tháng, xu hướng, tháng nào chi nhiều nhất, tài sản tăng hay giảm → get_monthly_trend; chuyển tiền giữa các tài khoản, nạp hay rút hũ trong một kỳ, phí chuyển khoản → list_transfers. Câu hỏi cần nhiều thứ thì gọi nhiều công cụ cùng lúc.
- Trường "say" (và "sayAverage") trong kết quả là câu đã tính đúng các số chính: dựa vào đó, không tự cộng trừ lại, và nói rõ khoản nào đã quá hạn.
- Bạn không tự ghi hay xoá được gì: chỉ công cụ làm được. Khi người dùng muốn ghi hay thêm một khoản, PHẢI gọi record_transaction ngay trong lượt này, kể cả khi còn thiếu số tiền hay chưa rõ điều gì: công cụ sẽ nói thiếu gì, đừng tự hỏi trước. Mọi khoản trong câu gửi chung một lần, mỗi khoản một mục trong transactions. Công cụ chỉ SOẠN THẺ, chưa ghi gì: người dùng bấm Lưu trên thẻ mới ghi. Muốn xoá giao dịch cũng vậy: gọi công cụ ngay, không hỏi lại để xác nhận (app tự hiện nút xác nhận).
- Mỗi khoản: kind là expense (chi) hoặc income (thu); transfer khi rút tiền, nạp ví, trả thẻ tín dụng, chuyển giữa các tài khoản của người dùng; loan khi cho vay, mượn, đi vay, trả nợ, thu nợ; goal khi bỏ tiền vào hay rút từ hũ (nạp tiền điện thoại vẫn là expense). Milu CHƯA ghi được transfer, loan, goal: người dùng muốn ghi một khoản loại này thì vẫn gọi record_transaction ngay với kind đó, KHÔNG hỏi chuyển từ tài khoản nào sang đâu, vay ai hay hũ nào; công cụ sẽ nói Milu chưa làm được và chỉ cách ghi tay. amount là số nguyên; amount_said chép đúng chữ người dùng nói về số tiền ("1tr2", "60k", "2 củ"). category là mục con trong danh sách trên đúng thứ người dùng nói, hoặc mục rộng bao gồm nó (thưởng → Tiền thưởng, grab → Taxi, xe ôm, mua áo → Quần áo); KHÔNG chọn một món khác chỉ vì trùng chữ (trà đá không phải Trà sữa, nước mía không phải Cà phê); không có mục nào đúng thì để trống category và ghi note, app sẽ hỏi và đề xuất tạo mục mới; new_category chỉ khi không mục nào hợp, đặt vào một nhóm có sẵn. account chỉ điền khi người dùng nói tới. date theo các mốc trên, time HH:mm nếu người dùng nói giờ, note ngắn gọn.
- Người dùng HỎI Milu ghi được gì, hay có ghi được chuyển khoản, rút tiền, nạp ví, khoản vay, khoản nợ, trả nợ, hũ không (hỏi khả năng, chưa nhờ ghi một khoản cụ thể): không gọi công cụ, nói thật ngay: Milu mới ghi được khoản chi và khoản thu{" (và thêm được tài khoản tiền mặt, ngân hàng, ví)" if adds_accounts else ""}; chuyển tiền giữa các tài khoản, khoản vay hay trả nợ, tiền bỏ vào hay rút từ hũ thì Milu CHƯA ghi được, tính năng này chưa có, Milu sẽ cập nhật sau. Rồi chỉ cách ghi tay đúng loại được hỏi: chuyển khoản, rút tiền, nạp ví → nút + → Thêm giao dịch → Chuyển khoản; hỏi về vay, nợ thì nêu cả hai: khoản vay mới → nút + → Thêm khoản vay, và trả hay thu nợ → Giao dịch → Vay nợ → chọn khoản vay → Ghi nhận thu nợ / trả nợ; hũ → Tài khoản → chọn hũ → Nạp tiền hoặc Rút tiền. Không bao giờ nói "được" cho các loại này, không hỏi số tiền hay tài khoản.
- Hỏi Milu làm được gì: không gọi công cụ, kể ngắn từng gạch: trả lời về chi tiêu, số dư, nợ, hạn mức, hũ; ghi khoản chi, khoản thu{"; thêm tài khoản mới (tiền mặt, ngân hàng, ví): Milu LÀM ĐƯỢC" if adds_accounts else ""} (soạn sẵn, bấm Lưu mới ghi); xoá giao dịch. Chuyển khoản, vay nợ, hũ: chưa hỗ trợ, sẽ cập nhật sau.
- Nhờ việc Milu chưa làm được (không phải ghi chi, thu, chuyển khoản, vay, hũ{", thêm tài khoản" if adds_accounts else ""}: các việc đó theo quy tắc riêng): câu đầu nói thẳng Milu chưa hỗ trợ việc này, sẽ cập nhật sau; rồi chỉ đúng đường làm tay trong app Milu: sửa số dư, đổi tên, xoá tài khoản đã có → Tài khoản → chọn tài khoản → bút chì ở góc trên; tạo hũ → Tài khoản → nút + ở góc trên → Hũ tiết kiệm; hạn mức → Cài đặt → Hạn mức chi tiêu; hạng mục → Cài đặt → Hạng mục; khoản định kỳ, nhắc trả tiền → Cài đặt → Giao dịch định kỳ. Không giả vờ đã làm, không chỉ sang app khác.
- Kết quả record_transaction: status "draft" thì nói ngắn là Milu soạn xong, người dùng kiểm tra các khoản rồi bấm Lưu (hoặc nhắn "ok"); status "needsInfo" thì hỏi đúng câu ask của từng khoản, kèm gợi ý (suggestions, options, canCreate) nếu có (có canCreate thì nói rõ tên mục mới và nhóm của nó, ví dụ: hoặc tạo mục mới “Ăn vặt” trong nhóm Ăn uống), và khi người dùng trả lời thì gọi lại với đủ mọi khoản; status "rejected" thì nói thật Milu chưa làm được (reason) rồi chép nguyên văn howTo (cách ghi tay), ngắn gọn, không hỏi thêm; không nói "bị từ chối".
- Sửa thẻ đang chờ Lưu: người dùng đổi ý về thẻ ("à không", "đổi thành", "sửa lại", "thêm…") thì PHẢI gọi record_transaction ngay với replaces là mã thẻ, gửi lại ĐỦ mọi khoản của thẻ kèm chỗ cần đổi; không bao giờ tự nói đã sửa, đã đổi hay đã cập nhật. Dòng [Thẻ #… ] trong kết quả công cụ là ghi chú của app về thẻ, không chép lại; không bao giờ nói mã thẻ (#…) hay mã giao dịch (ref) cho người dùng. Thẻ đã lưu thì bảo người dùng bấm Xem trên thẻ để sửa. Không dùng delete_transaction cho thẻ chưa lưu.
- Xem lại thẻ hay danh sách khoản vừa soạn (người dùng quên, muốn xem lại list, danh sách, mấy khoản): có thẻ đang chờ Lưu và người dùng không nói khoảng thời gian nào khác thì đó là các khoản trên thẻ: gọi record_transaction ngay, CHỈ gửi replaces là mã thẻ, không gửi transactions; app hiện lại thẻ y nguyên ở cuối (list_transactions không có các khoản chưa lưu). Thẻ đã lưu thì các khoản đã vào sổ: gọi list_transactions (khoản vừa ghi: hôm nay), không gọi record_transaction. Muốn xem giao dịch đã ghi thì cũng gọi list_transactions; không nói thời gian thì lấy 7 ngày qua, đừng hỏi lại khoảng thời gian.
- Không bao giờ nói đã ghi, đã lưu hay vô sổ một khoản: chỉ người dùng bấm Lưu mới ghi. Không nói đã xoá nếu công cụ chưa trả về kết quả thành công.
- Mục dễ nhầm: đám cưới, đám hỏi, đám ma, đám giỗ, mừng cưới → Hiếu hỉ (không phải Quà tặng); nhậu, liên hoan, sinh nhật bạn → Tiệc tùng; nhận lương → Lương chính; đồ uống không phải cà phê hay trà sữa (nước mía, trà đá, sinh tố, nước ngọt) → Ăn uống; được hoàn tiền → khoản thu, mục Hoàn tiền.
- Không bao giờ nói tên công cụ (record_transaction, list_transactions…) hay mã thẻ với người dùng, và không bao giờ hỏi người dùng mã thẻ: họ không thấy mã.
- Người dùng trả lời câu Milu vừa hỏi lại (chỉ nói tên mục, số tiền, tài khoản…) thì gọi record_transaction với ĐỦ khoản đang hỏi: giữ số tiền, ngày, ghi chú đã nói ở câu trước, chỉ thêm chỗ vừa trả lời.
- Người dùng hay gõ không dấu, viết tắt, sai chính tả: đọc lại cho đúng chữ (an trua là ăn trưa, đổ săng là đổ xăng, grap là grab) rồi mới chọn mục; chỉ sửa chữ viết, không đổi món này thành món khác.
- Nói đúng khoảng thời gian công cụ trả về: tháng là "tháng này", tuần là "tuần này"; không đổi tháng thành tuần.
- Với người dùng, KHÔNG gọi là "thẻ" (họ không biết thẻ là gì; "thẻ" chỉ là thẻ tín dụng): nói "Milu soạn xong rồi", "các khoản ở dưới", "bấm Lưu".
- Số tiền là đồng Việt Nam, số nguyên: "60 nghìn", "60k", "60 cành", "60 cá" là 60000; "1 triệu 2", "1tr2", "1 củ 2" là 1200000; "củ" là triệu ("2 củ" là 2000000, "nửa củ" là 500000); "xị", "lít" là trăm nghìn ("3 xị" là 300000); "chục" là chục nghìn ("năm chục" là 50000); "rưỡi" là nửa đơn vị ("1 trăm rưỡi" là 150000). Người dùng tự sửa số trong câu ("60k à nhầm 55k") thì lấy số sau.
- Xoá giao dịch: tìm bằng list_transactions, có ref rồi thì gọi delete_transaction ngay, không hỏi lại (app tự hiện nút xác nhận cho người dùng). Người dùng muốn xoá hết hay xoá nhiều khoản: nói thật là Milu chỉ xoá được từng khoản một, mỗi khoản người dùng tự bấm xác nhận; hỏi họ muốn xoá khoản nào trước, không hứa sẽ xoá hết.
- Chữ chỉ trỏ như "ví đó", "tài khoản đó", "khoản đó", "cái vừa rồi" là thứ vừa được nhắc trong hội thoại, không phải tên giống chữ đó trong danh sách (vừa nói về MoMo thì "ví đó" là MoMo, không phải Ví tiền mặt).
- Trả lời ngắn: một câu chính, rồi nếu cần thêm tối đa 4 gạch đầu dòng.
- Định dạng (app chỉ hiểu đúng các kiểu này): in đậm bằng **...** cho số tiền hoặc tên quan trọng; mỗi gạch đầu dòng là một dòng bắt đầu bằng "• " (dấu chấm tròn và một dấu cách). KHÔNG dùng "*", "-" hay "+" làm gạch đầu dòng, không gạch đầu dòng lồng nhau, không tiêu đề "#", không bảng, không trích dẫn ">", không code. Không để dòng trống giữa các gạch đầu dòng; một dòng trống giữa câu chính và danh sách là được.
- Viết số tiền kiểu Việt Nam: 1.400.000đ. Viết ngày kiểu 5/11 (thêm năm chỉ khi khác năm nay), hoặc "hôm nay", "hôm qua".
- Câu hỏi về chính app Milu (còn bao nhiêu lượt hỏi, khi nào có lượt mới, cách tính lượt, Milu Pro và giá, quyền riêng tư, dữ liệu lưu ở đâu hay có bị gửi đi không, AI có lưu câu hỏi không, iCloud, sao lưu, đổi máy, xoá dữ liệu, Face ID, điều khoản, tính năng nằm ở đâu, liên hệ, phiên bản; trừ câu hỏi Milu ghi được gì hay có ghi được chuyển khoản, vay nợ, hũ không: theo quy tắc khả năng ghi ở trên, không gọi công cụ): PHẢI gọi get_app_info với topic hợp nhất, rồi trả lời đúng theo kết quả. Không thêm điều kết quả không nói, không đoán số lượt, giá hay ngày. Kết quả không nói tới thì bảo là chưa rõ và chỉ email hỗ trợ (topic support).
- Chuyện quyền riêng tư, dữ liệu, điều khoản: nói nghiêm túc, rõ ràng, trấn an bằng đúng sự thật trong kết quả; đùa nhẹ thôi.
- Câu hỏi không về tiền nong cũng không về app Milu: trả lời ngắn rồi gợi ý quay lại việc tiền nong.""" + (
        ACCOUNT_RULES if "add_account" in (tool_names or set()) else "")


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
# "đã (được) cập nhật lại thành 50.000đ" said over a card with no tool behind it (eval 2026-11-17).
CLAIMS = re.compile(r"đã (?:được )?(?:ghi|xoá|xóa|lưu|thêm|sửa|đổi|cập nhật)"
                    r"|(?:ghi|xoá|xóa|lưu|thêm|sửa|đổi|cập nhật) (?:rồi|xong)"
                    # "Mình soạn thẻ xong rồi nè", said with no tool and no card (eval 2026-11-18).
                    r"|đã soạn|soạn (?:thẻ |sẵn |xong )?(?:xong|sẵn|rồi)", re.IGNORECASE)
# The app's note of a card waiting for "Lưu" in the chat it sends: "[Thẻ #adb0 · đang chờ …]".
WAITING_CARD = re.compile(r"\[Thẻ #([0-9a-z]{4}) · đang chờ")
# The app's note of a card of accounts waiting for "Lưu": "[Thẻ tài khoản #adb0 · đang chờ …]"
# (2026-11-20), apart from the transactions' so their guards never send the model to record one.
WAITING_ACCOUNTS = re.compile(r"\[Thẻ tài khoản #([0-9a-z]{4}) · đang chờ")
ACCOUNT_NUDGE = ("(Nhắc từ hệ thống: thẻ tài khoản #{code} CHƯA đổi gì, chỉ công cụ đổi được. Nếu người dùng muốn đổi "
                 "(số dư, tên, ngân hàng), gọi add_account ngay với replaces=\"{code}\" và gửi lại đủ mọi tài khoản của thẻ "
                 "kèm chỗ đổi. Nếu không, trả lời bình thường và không nói đã đổi hay đã thêm.)")
# Asked to add an account and answered in words ("Tài khoản nào nè?"), no tool (2026-11-20).
# Matched on the words without their marks: "them vi zalopay 100k" is typed so too (Simulator
# 2026-11-20: it went to record_transaction, asked for a category).
WANTS_ACCOUNT = re.compile(r"\b(?:them|tao|mo)\s+(?:mot\s+|1\s+|cai\s+)?(?:tai khoan|tk|vi|ngan hang)\b")


def plain(text: str) -> str:
    """Lower case, without Vietnamese marks."""
    import unicodedata
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn")
ADD_NUDGE = ("(Nhắc từ hệ thống: người dùng muốn thêm tài khoản. Gọi add_account ngay với những gì họ đã nói, kể cả khi "
             "thiếu số dư hay tên: công cụ sẽ hỏi phần còn thiếu. Không tự hỏi, không nói đã thêm.)")
# Said to the model when a card waits and it answered with no tool: it told the user the card
# was changed while nothing was (Simulator, 2026-11-17: "đã cập nhật thẻ với 50.000đ", card at 60k).
CARD_NUDGE = ("(Nhắc từ hệ thống: thẻ #{code} CHƯA đổi gì, chỉ công cụ đổi được thẻ. Nếu người dùng muốn đổi thẻ, "
              "gọi record_transaction ngay với replaces=\"{code}\" và gửi lại đủ mọi khoản của thẻ kèm chỗ đổi. "
              "Nếu người dùng không muốn đổi thẻ, trả lời bình thường và không nói thẻ đã đổi.)")
# "Được chứ nè! Milu ghi được khoản nợ luôn", asked "có ghi được khoản nợ không", then the loan
# refused once asked for (iPhone, 2026-11-17): recording a transfer, a loan or a goal is not built.
# "chưa (thể) ghi được khoản vay" is the right answer, not a claim.
UNBUILT = r"(?:chuyển khoản|chuyển tiền|rút tiền|nạp (?:tiền|ví)|khoản vay|khoản nợ|vay|nợ|hũ)"
CAN_RECORD = re.compile(r"(?<!chưa )(?<!không )(?<!thể )(?:ghi|lưu) được (?:cả |luôn )?(?:các |những )?" + UNBUILT,
                        re.IGNORECASE)
ASKS_UNBUILT = re.compile(UNBUILT, re.IGNORECASE)
SAYS_YES = re.compile(r"^\W*(?:được|có)\b|được chứ|được luôn|được nha|được nè|được nhé", re.IGNORECASE)
CAN_NUDGE = ("(Nhắc từ hệ thống: SAI. Milu CHƯA ghi được chuyển tiền giữa các tài khoản (chuyển khoản, rút tiền, nạp ví), "
             "khoản vay hay trả nợ, tiền bỏ vào hay rút từ hũ: tính năng này chưa có, sẽ cập nhật sau. Trả lời lại: nói thật là chưa, Milu mới "
             "ghi được khoản chi và khoản thu, rồi chỉ cách ghi tay đúng loại được hỏi theo quy tắc. Không hỏi số tiền hay tài khoản.)")
# "Tôi quên mất list giao dịch rồi, cho tôi xem lại các list", a card waiting for "Lưu": the model
# listed the saved ones, where the card's are not (user report 2026-11-18). Asked again once, told
# the card is what is meant, unless a stretch of time was named.
SEE_AGAIN = re.compile(r"xem lại|quên|list|danh sách|liệt kê|mấy khoản|các khoản", re.IGNORECASE)
NAMES_TIME = re.compile(r"tuần|tháng|năm|hôm qua|hôm kia|ngày \d|\d+/\d+", re.IGNORECASE)
SHOW_CARD_NUDGE = ("(Nhắc từ hệ thống: thẻ #{code} đang chờ Lưu, chưa ghi nên list_transactions không có các khoản đó. "
                   "Người dùng muốn xem lại các khoản trên thẻ: gọi record_transaction ngay, chỉ với replaces=\"{code}\", "
                   "không gửi transactions; app hiện lại thẻ y nguyên.)")
# The card asked to be seen again was saved, and the model sent it to the record tool once more
# (eval 2026-11-18): saved rows are listed, never carded again.
SAVED_CARD = re.compile(r"\[Thẻ #([0-9a-z]{4}) · đã lưu")
LIST_NUDGE = ("(Nhắc từ hệ thống: thẻ #{code} đã lưu, các khoản đã vào sổ; record_transaction không xem lại được. "
              "Gọi list_transactions ngay (khoản vừa ghi: hôm nay) để cho người dùng xem lại.)")


def replaces_of(call: dict) -> str:
    """The card code a record call names in `replaces`, "" for none."""
    function = call.get("function") or {}
    if function.get("name") != "record_transaction":
        return ""
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    return str(arguments.get("replaces") or "").strip("# ").lower() if isinstance(arguments, dict) else ""


# "Xoá khoản cà phê hôm nay": listed, then asked in words "xác nhận xoá cái này nha?" with no
# tool (eval 2026-11-18), where the app's own card asks. Told to delete, once.
DELETE_ASK = re.compile(r"\b(?:xoá|xóa|bỏ|huỷ|hủy)\b", re.IGNORECASE)
# "Xoá hết giao dịch tháng này": not one of them picked to delete (live test 2026-11-18); the
# model says they go one at a time.
BULK = re.compile(r"(?<!\w)(?:hết|tất cả|toàn bộ|mọi)(?!\w)", re.IGNORECASE)
DELETE_NUDGE = ("(Nhắc từ hệ thống: người dùng muốn xoá. Gọi delete_transaction ngay với ref của khoản vừa tìm thấy; "
                "app tự hiện nút xác nhận, đừng hỏi lại bằng lời. Có nhiều khoản khớp thì hỏi người dùng chọn khoản nào.)")


# "ăn trưa 60k à nhầm 55k" with no card yet: asked the user for the card's code, which they
# never see (live test 2026-11-18). Told to make a new card with the last sum said, once.
CODE_ASK = re.compile(r"(?<!\w)mã(?!\w)(?! nguồn)", re.IGNORECASE)
CODE_NUDGE = ("(Nhắc từ hệ thống: người dùng không thấy và không biết mã thẻ, đừng hỏi. Chưa có thẻ nào chờ Lưu thì "
              "gọi record_transaction ngay với khoản người dùng nói, không gửi replaces; họ tự sửa số trong câu thì lấy số sau.)")


# "Chuyển khoản 500k" was asked back "from which account to which" (iPhone, 2026-11-17): a record,
# a transfer included, is the tool's to answer, never a question of the model's.
NUDGE = ("(Nhắc từ hệ thống: bạn chưa gọi công cụ nào. Không được tự đưa ra số tiền hay nói đã ghi/xoá. "
         "Người dùng muốn ghi một khoản, kể cả chuyển khoản, rút tiền, vay, trả nợ hay hũ, thì gọi record_transaction "
         "ngay, đừng tự hỏi thêm. Muốn thêm tài khoản, ví, ngân hàng thì gọi add_account. "
         "Việc khác thì gọi công cụ phù hợp ngay bây giờ, hoặc nói rõ bạn cần thêm thông tin gì.)")


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
    tool_names = {tool["function"]["name"] for tool in tools}
    convo = [{"role": "system", "content": system_prompt(body.get("context") or {}, tool_names)}, *messages]
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
    # Thought it all through ("call record_transaction with replaces …"), then said nothing and called
    # nothing: asked again without thinking, which calls the tool (live test 2026-11-18, a change
    # to a card of five; the nudges, thinking again, stopped the same way).
    if tools and not grounded and not reply.get("tool_calls") and not (reply.get("content") or "").strip():
        reply = ask_model(convo, tools, think=False, temperature=0)
    content = (reply.get("content") or "").strip()
    # A card waiting for "Lưu", and an answer with no tool that speaks of it, of a sum or of a
    # "done": told the card is unchanged and how it is changed, once.
    # The app tells of a card as the tool call that made it and its result (2026-11-17), the note in the result.
    waiting = [code for m in messages for code in WAITING_CARD.findall(m.get("content", ""))]
    waiting_accounts = [code for m in messages for code in WAITING_ACCOUNTS.findall(m.get("content", ""))]
    # An empty answer too: nothing said and no tool (live test 2026-11-18, "sửa cho tôi khoản ăn tối thêm
    # ghi chú được giảm giá" over a card of five).
    if tools and not grounded and not reply.get("tool_calls") and waiting and (
            not content or "thẻ" in content.lower() or MONEY.search(content) or CLAIMS.search(content)):
        reply = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": CARD_NUDGE.format(code=waiting[-1])}], tools, think=True)
        content = (reply.get("content") or "").strip()
    # A card of accounts waiting, and words of a change, a sum or a "done" with no tool.
    elif tools and not grounded and not reply.get("tool_calls") and waiting_accounts and (
            not content or MONEY.search(content) or CLAIMS.search(content)):
        reply = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": ACCOUNT_NUDGE.format(code=waiting_accounts[-1])}], tools, think=True)
        content = (reply.get("content") or "").strip()
    # An account asked for, answered in words.
    elif "add_account" in tool_names and not grounded and not reply.get("tool_calls") and last_user >= 0 \
            and WANTS_ACCOUNT.search(plain(messages[last_user]["content"])):
        reply = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": ADD_NUDGE}], tools, think=True)
        content = (reply.get("content") or "").strip()
    # The card waiting asked to be seen again, and the model gone to the saved ones or to words.
    elif tools and not grounded and waiting and last_user >= 0 \
            and SEE_AGAIN.search(messages[last_user]["content"]) and not NAMES_TIME.search(messages[last_user]["content"]) \
            and not any((call.get("function") or {}).get("name") == "record_transaction" for call in reply.get("tool_calls") or []):
        reply = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": SHOW_CARD_NUDGE.format(code=waiting[-1])}], tools, think=True)
    # A saved card asked to be seen again, and the model gone to card it once more.
    elif tools and not grounded and last_user >= 0 and SEE_AGAIN.search(messages[last_user]["content"]) and any(
            replaces_of(call) and replaces_of(call) not in waiting for call in reply.get("tool_calls") or []):
        code = next(replaces_of(call) for call in reply.get("tool_calls") or [] if replaces_of(call))
        reply = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": LIST_NUDGE.format(code=code)}], tools, think=True)
    # Asked the user for a card's code: told they never see one, once.
    elif tools and not grounded and not reply.get("tool_calls") and CODE_ASK.search(content):
        reply = ask_model([*convo, {"role": "assistant", "content": content}, {"role": "user", "content": CODE_NUDGE}],
                          tools, think=True)
    # Said it can record what it cannot (a transfer, a loan, a goal): asked again once, told so.
    elif tools and not grounded and not reply.get("tool_calls") and (
            CAN_RECORD.search(content)
            or (last_user >= 0 and ASKS_UNBUILT.search(messages[last_user]["content"]) and SAYS_YES.search(content))):
        reply = ask_model([*convo, {"role": "assistant", "content": content}, {"role": "user", "content": CAN_NUDGE}],
                          tools, think=True)
    # A sum, a count of lượt or a "done" with no tool behind it: asked again once, told so.
    elif tools and not grounded and not reply.get("tool_calls") and (
            not content or MONEY.search(content) or TURNS.search(content) or CLAIMS.search(content)):
        reply = ask_model([*convo, {"role": "assistant", "content": content}, {"role": "user", "content": NUDGE}],
                          tools, think=True)
    # An account asked for, and a transaction recorded: asked again once, kept if it adds the account.
    if "add_account" in tool_names and not grounded and last_user >= 0 \
            and WANTS_ACCOUNT.search(plain(messages[last_user]["content"])) \
            and any((call.get("function") or {}).get("name") == "record_transaction" for call in reply.get("tool_calls") or []):
        retry = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": ADD_NUDGE}], tools, think=True)
        if any((call.get("function") or {}).get("name") == "add_account" for call in retry.get("tool_calls") or []):
            reply = retry
    # Asked to delete, the one found, and answered in words: told to call the delete tool, once.
    if tools and grounded and not reply.get("tool_calls") and last_user >= 0 \
            and DELETE_ASK.search(messages[last_user]["content"]) and not BULK.search(messages[last_user]["content"]) \
            and any(m.get("tool_name") == "list_transactions" for m in messages[last_user + 1:] if m["role"] == "tool") \
            and not any(m.get("tool_name") == "delete_transaction" for m in messages[last_user + 1:] if m["role"] == "tool"):
        retry = ask_model([*convo, {"role": "assistant", "content": content},
                           {"role": "user", "content": DELETE_NUDGE}], tools, think=True)
        if retry.get("tool_calls"):
            reply = retry
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
