# milu-server

Máy chủ AI cho app iOS **Milu**: nhận dạng giọng nói tiếng Việt và LLM cục bộ, sau một API.

```
iPhone (Milu) → https://milu-api.nnqlab.dev → Cloudflare Tunnel → api ─┬→ speech (sherpa-onnx, CPU)
                                                                      └→ llm (Ollama gemma, GPU)
```

Dữ liệu của người dùng ở trên điện thoại. LLM không đọc kho dữ liệu: app gửi câu hỏi kèm danh sách
**tool** (tên, mô tả, tham số JSON Schema), model chọn tool, **app tự chạy** tool trên dữ liệu trong máy
rồi gửi lại kết quả, model viết câu trả lời. Thêm khả năng mới = thêm tool trong app
(`Milu/Features/Assistant/Services/AssistantTools.swift`), không cần sửa máy chủ.

## API

| | |
|---|---|
| `GET /health` | `{ok, model}` |
| `POST /v1/chat` | `{messages, tools, context}` → `{message: {content, toolCalls}, ms}` |
| `POST /v1/transcribe` | thân = file ghi âm (M4A), header `X-Hotwords` (tên, URL-encode, ngăn bằng `/`) → `{text, ms}` |

Mọi `/v1` cần `Authorization: Bearer <MILU_API_KEYS>`, giới hạn theo giờ cho mỗi khoá. Không ghi log
nội dung, chỉ đường dẫn, mã trả về và thời gian.

Khi chọn tool, gemma bật chế độ suy nghĩ (không bật thì nó tự bịa số); khi viết câu trả lời từ kết quả
tool thì tắt cho nhanh. Nếu model đưa số tiền hoặc nói "đã ghi" mà chưa gọi tool, `api` hỏi lại một lần.

## Chạy

```sh
cp .env.example .env   # điền khoá, tunnel id; chmod 600 .env
docker compose up -d --build
docker compose logs -f api
```

## Ollama

GPU chỉ đủ một bản model. Khi web Timi (project `finance`) còn chạy, `api` dùng Ollama của nó qua
`compose.shared-llm.yaml`. Ngày tắt Timi, sửa `.env` theo `.env.example` (`COMPOSE_FILE=compose.yaml`,
`COMPOSE_PROFILES=llm`, `OLLAMA_URL=http://llm:11434`) rồi `docker compose up -d`. Model đã tải nằm
trong volume `finance-ollama`, `llm` dùng lại.

## Chấm điểm

`eval/cases.json` là bộ câu hỏi mẫu kèm đáp án (tool phải gọi, tham số, và câu trả lời phải có
gì). `eval/app.json` là danh sách tool và ngữ cảnh đúng như app gửi; xuất lại mỗi khi sửa tool
trong app:

```sh
TEST_RUNNER_MILU_EXPORT=$PWD/eval/app.json xcodebuild test ... -only-testing:MiluTests/AssistantToolsTests
```

Chạy (mỗi câu 3 lượt, kết quả ở `eval/results.md`):

```sh
MILU_API_KEY=... python3 eval/run.py --runs 3
```

Nên chạy lại mỗi khi sửa lời dặn trong `api/server.py`, đổi model, hoặc thêm tool.
