# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature = 0.0, câu trả lời mang tính nhất quán tuyệt đối, tập trung vào thông tin thực tế cơ bản và dự đoán token có xác suất cao nhất. Khi tăng lên 0.5 – 1.0, câu từ trở nên sinh động, phong phú hơn và cách diễn đạt linh hoạt hơn. Ở temperature = 1.5, phản hồi xuất hiện nhiều từ ngữ bất ngờ, sự sáng tạo tăng cao nhưng có nguy cơ bắt đầu mâu thuẫn hoặc thiếu chính xác (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt temperature từ 0.0 đến 0.2 cho chatbot hỗ trợ khách hàng. Lý do là hệ thống CSKH ưu tiên hàng đầu sự chính xác, nhất quán và đáng tin cậy của thông tin (như chính sách hoàn tiền, giá sản phẩm, quy trình dịch vụ), tránh việc model tự sáng tạo thông tin sai lệch gây nhầm lẫn cho khách hàng.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần cho workload này (khoảng $105/ngày so với $6.30/ngày). Trường hợp xứng đáng dùng GPT-4o là khi cần lập luận logic phức tạp, viết mã nguồn nâng cao hoặc phân tích tài liệu pháp lý chuyên sâu. Trường hợp nên dùng mini là các bài toán như tóm tắt văn bản ngắn, phân loại ý định (intent classification) hoặc trả lời câu hỏi CSKH đơn giản.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi dành cho học sinh tiểu học sử dụng từ ngữ đơn giản, độ dài vừa phải và các hình ảnh so sánh đời thường (như sổ nhật ký chung của lớp học). Ngược lại, phản hồi của chuyên gia tài chính đi thẳng vào bản chất kỹ thuật với các thuật ngữ như 'sổ cái phân tán' (distributed ledger), 'mã hóa phi tập trung' (decentralized cryptography), và 'cơ chế đồng thuận' (consensus mechanism). System prompt đóng vai trò như định hướng vai trò (persona), điều chỉnh giọng văn, độ sâu tri thức và cách tiếp cận phù hợp với từng đối tượng mục tiêu.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt ~100 từ (ước tính thô ~133 token), `count_tokens` (tiktoken) trả về khoảng 160–180 token, chênh lệch khoảng 20% – 35% so với công thức ước lượng thô. Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì bộ mã hóa (tokenizer) của OpenAI được huấn luyện chủ yếu trên dữ liệu tiếng Anh, do đó các từ tiếng Việt có dấu thường bị tách thành nhiều sub-word hoặc byte-level token nhỏ hơn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng chatbot tương tác trực tiếp với người dùng (như ChatGPT, AI Assistant), nơi phản hồi dài mất nhiều giây để sinh ra; việc hiển thị ngay từng từ giúp giảm Perceived Latency (thời gian chờ đợi cảm nhận được) và tạo trải nghiệm mượt mà. Ngược lại, non-streaming phù hợp hơn cho các tiến trình xử lý ngầm (background jobs), gọi API giữa các service (backend-to-backend), hoặc khi cần nhận trọn vẹn dữ liệu định dạng JSON/structured output để parse trước khi xử lý tiếp.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giãn cách thời gian giữa các lần thử lại (0.1s -> 0.2s -> 0.4s...), tạo điều kiện cho hệ thống server có thời gian phục hồi khi đang bị nghẽn mạng hoặc quá tải. Nếu hàng nghìn client cùng retry với delay cố định giống nhau (thảm họa thundering herd problem), tất cả các request sẽ đồng thời nã vào server ở cùng một thời điểm, tiếp tục làm sập hệ thống và khiến tỷ lệ thất bại lên tới 100%.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona: "Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt." Lựa chọn từ ngữ: "ngắn gọn" giúp hạn chế mô hình sinh phản hồi dài dòng không cần thiết, giảm lượng token output và tiết kiệm chi phí; chỉ định "tiếng Việt" đảm bảo mô hình luôn trả lời đúng ngôn ngữ người dùng yêu cầu dù prompt có chứa thuật ngữ tiếng Anh.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất hiện tại của trợ lý là bộ nhớ lịch sử hội thoại ngắn (chỉ lưu 3 lượt gần nhất `history[-6:]`), khiến mô hình quên thông tin nếu cuộc trò chuyện kéo dài. Giải pháp cải thiện là triển khai kỹ thuật Tóm tắt lịch sử (Conversation Summary Buffer) hoặc RAG / Vector Database: định kỳ dùng model nhỏ để tóm tắt các lượt hội thoại cũ thành một bản tóm tắt ngắn gọn và chèn bản tóm tắt đó vào system prompt.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
