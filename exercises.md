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
> Chạy 5 lần ở mỗi mức (0.0 / 0.5 / 1.0 / 1.5) thì cả 20 phản hồi đều khác nhau
> về câu chữ, **kể cả ở temperature = 0.0**. Điều này không mâu thuẫn với
> README — README chỉ nói temperature thấp "thường ổn định và dễ lặp lại **hơn**"
> và nhấn mạnh "đây là xu hướng chứ không phải cam kết tuyệt đối". Nó chỉ mâu
> thuẫn với cách hiểu phổ biến rằng temperature = 0 tương đương greedy decoding
> nên phải cho ra kết quả y hệt nhau. Về mặt công thức thì đúng là vậy, nhưng
> phía phục vụ vẫn còn nguồn ngẫu nhiên khác: thứ tự cộng dồn số thực thay đổi
> theo cách gom batch, định tuyến expert của kiến trúc MoE phụ thuộc vào các
> request đi cùng batch, cộng với tính toán ở độ chính xác hỗn hợp. Chỉ cần
> logit xê dịch rất nhỏ là token đứng đầu bị đảo khi hai ứng viên sát nhau.
>
> Xét theo *nội dung* thay vì câu chữ: 0.5 và 1.0 đều kể về hang Sơn Đoòng cả
> 5/5 lần, còn 0.0 và 1.5 trộn giữa Sơn Đoòng và cà phê Robusta. Với n = 5 thì
> mình **không** quan sát được xu hướng đơn điệu "temperature càng cao càng đa
> dạng" — cỡ mẫu này quá nhỏ để tách tín hiệu khỏi nhiễu, nên đây là hạn chế
> của phép đo chứ chưa đủ để kết luận xu hướng đó sai. Cái thấy rõ hơn là
> temperature tác động lên *cách diễn đạt* nhiều hơn lên *việc chọn chủ đề*,
> vì model đã được post-train nên có sẵn một câu trả lời "tủ". Riêng ở 1.5 có
> một lần model gõ sai chính tả ("sơn độong") — dấu hiệu chất lượng bắt đầu
> suy giảm khi lấy mẫu quá rộng.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình đặt thấp, khoảng **0.0 – 0.3**. Chatbot hỗ trợ khách hàng cần cùng một
> câu hỏi ra cùng một câu trả lời: để QA viết được test case, để nhân viên đối
> chiếu khi khách khiếu nại, và để giảm rủi ro model bịa ra chính sách hoặc giá
> không tồn tại. Tính sáng tạo ở đây không mang lại giá trị, trong khi một câu
> trả lời sai về điều khoản hoàn tiền thì tốn tiền thật.
>
> Nhưng đúng theo những gì đo được ở Câu 1.1, temperature thấp chỉ **giảm** chứ
> không **loại bỏ** được biến thiên. Nên với nhóm câu hỏi nhạy cảm (giá, chính
> sách, pháp lý) vẫn phải trả bằng nội dung soạn sẵn hoặc RAG có trích nguồn,
> chứ không phó mặc cho model sinh tự do ở bất kỳ temperature nào.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính gemini-3.5-flash-lite đắt hơn gemini-3.1-flash-lite bao nhiêu lần cho
workload này? Nêu một trường hợp model chính xứng đáng với chi phí và một
trường hợp nên dùng model nhỏ:**
> **Tính khối lượng:** 10.000 người × 3 lượt = 30.000 lượt gọi/ngày.
> 30.000 × 350 token đầu ra = **10.500.000 token output/ngày** = 10.500 đơn vị 1K.
>
> | Model | Giá output /1K | Chi phí/ngày | Chi phí/tháng (30 ngày) |
> |---|---|---|---|
> | `gemini-3.5-flash-lite` | $0,0025 | **$26,25** | $787,50 |
> | `gemini-3.1-flash-lite` | $0,0015 | **$15,75** | $472,50 |
>
> **Đắt hơn 1,67 lần** (0,0025 / 0,0015), chênh $10,50/ngày ≈ **$315/tháng**.
> Tỷ lệ này chỉ tính token output; nếu tính cả input thì chênh lệch nhỏ lại
> (input $0,30 so với $0,25 /1M, tức chỉ 1,2 lần).
>
> **Khi model chính xứng đáng:** những tác vụ mà một câu sai tốn nhiều hơn
> $315/tháng — tóm tắt hợp đồng, sinh code đưa thẳng vào sản phẩm, suy luận
> nhiều bước, hoặc nội dung hiển thị cho khách mà không có người duyệt lại.
> Ở đó chi phí sửa sai lớn hơn hẳn chi phí token.
>
> **Khi nên dùng model nhỏ:** đúng như đo được ở Checkpoint 1 với câu "Việt Nam
> có bao nhiêu tỉnh thành" — cả hai model đều trả lời đúng, model nhỏ còn nhanh
> gấp 2,2 lần (0,95s so với 2,11s). Với tra cứu đơn giản, phân loại, định tuyến
> intent hay trích xuất trường dữ liệu thì model nhỏ vừa rẻ vừa cho trải nghiệm
> tốt hơn nhờ độ trễ thấp.
>
> *Lưu ý: bảng giá trong `template.py` là giá paid tier dùng để học. Cả hai model
> đều có free tier, nhưng 30.000 lượt/ngày thì vượt hạn mức miễn phí.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Câu trả lời của bạn*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *Câu trả lời của bạn*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Câu trả lời của bạn*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Câu trả lời của bạn*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Câu trả lời của bạn*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Câu trả lời của bạn*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
