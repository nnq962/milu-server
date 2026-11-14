# Kết quả chấm điểm Milu (2026-11-15 06:56)

**210/214 lượt đạt (98%)**, 107 câu × 2 lượt. Thời gian trung vị một lượt gọi: 1092 ms.

| Nhóm | Đạt | Tỉ lệ |
|---|---|---|
| đọc | 32/32 | 100% |
| ghi | 25/26 | 96% |
| xoá | 4/4 | 100% |
| trò chuyện | 8/8 | 100% |
| nối tiếp | 2/4 | 50% |
| không dấu | 8/8 | 100% |
| câu trả lời | 12/12 | 100% |
| về app | 40/40 | 100% |
| trả lời về app | 26/26 | 100% |
| tài chính | 44/44 | 100% |
| trả lời tài chính | 9/10 | 90% |

## Câu chưa đạt

- **w11** `ghi 50k` — 1/2: không gọi tool (cần record_transaction); trả lời: 'Ủa, **50k** là chi hay thu vậy bạn? Với lại là chi cho cái gì nè? 💸'<br>Mẫu: `Ủa, **50k** là chi hay thu vậy bạn? Với lại là chi cho cái gì nè? 💸`
- **f02** `ghi thêm 20k gửi xe bằng ví đó` — 0/2: account='Ví tiền mặt' (cần {'names': ['MoMo']})<br>Mẫu: `{"id": "call_h7a7y48q", "name": "record_transaction", "arguments": {"account": "Ví tiền mặt", "amount": 20000, "category": "Gửi xe", "kind": "expense", "note": `
- **u03** `trung bình mỗi tháng mình tiêu bao nhiêu` — 1/2: thiếu một trong ['2 tháng', 'hai tháng']<br>Mẫu: `Ủa, trung bình các tháng gần đây bạn tiêu khoảng **16.000.000đ** nha 💸.

• Tháng 10 là tháng bạn chi nhiều nhất với **18.000.000đ**.
• Tháng 9 là tháng bạn tiết`

## Ghi chú (không trừ điểm)

- **a03**: dài (9 dòng)
