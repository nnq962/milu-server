# Kết quả chấm điểm Milu (2026-11-12 11:22)

**137/141 lượt đạt (97%)**, 47 câu × 3 lượt. Thời gian trung vị một lượt gọi: 1263 ms.

| Nhóm | Đạt | Tỉ lệ |
|---|---|---|
| đọc | 48/48 | 100% |
| ghi | 37/39 | 95% |
| xoá | 6/6 | 100% |
| trò chuyện | 12/12 | 100% |
| nối tiếp | 4/6 | 67% |
| không dấu | 12/12 | 100% |
| câu trả lời | 18/18 | 100% |

## Câu chưa đạt

- **w04** `cà phê 45k trả bằng momo` — 2/3: lỗi gọi API: <urlopen error _ssl.c:1015: The handshake operation timed out><br>Mẫu: `{"id": "call_kyq1nah2", "name": "record_transaction", "arguments": {"account": "MoMo", "amount": 45000, "category": "Cà phê", "kind": "expense", "note": "Cà phê`
- **w11** `ghi 50k` — 2/3: không gọi tool (cần record_transaction); trả lời: 'Ủa, **50k** là chi hay thu vậy bạn? 💸 Với lại, chi cho cái gì nè? Cho Milu biết '<br>Mẫu: `Ủa, **50k** là chi hay thu vậy bạn? 💸 Với lại, chi cho cái gì nè? Cho Milu biết danh mục để ghi sổ nha! 😏`
- **f02** `ghi thêm 20k gửi xe bằng ví đó` — 1/3: account='Ví tiền mặt' (cần {'names': ['MoMo']})<br>Mẫu: `{"id": "call_gnoabf3g", "name": "record_transaction", "arguments": {"account": "MoMo", "amount": 20000, "category": "Gửi xe", "kind": "expense", "note": "Gửi xe`
