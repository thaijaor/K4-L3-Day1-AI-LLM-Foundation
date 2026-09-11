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
> Ở prompt này temperature chủ yếu đổi *cách diễn đạt*, nhưng *chủ đề* vẫn dao
> động giữa hai fact gần ngang xác suất (Sơn Đoòng và cà phê Robusta) — kể cả
> ở 0.0 (3/2), nên không thể nói "chỉ khác câu chữ". Đáng chú ý là 0.0 vẫn cho
> 5/5 phản hồi khác nhau, tức 0.0 không đồng nghĩa lặp lại y hệt. Với n = 5
> mình chưa thấy rõ "temp cao thì đa dạng hơn"; cỡ mẫu quá nhỏ để kết luận.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Đặt thấp, khoảng **0.0 – 0.3**: hỗ trợ khách hàng cần nhất quán và chính xác
> về chính sách/giá, không cần sáng tạo. Lưu ý temperature thấp chỉ giảm chứ
> không loại bỏ biến thiên (thấy ở Câu 1.1), nên câu về giá/pháp lý vẫn nên
> chốt bằng nội dung soạn sẵn hoặc RAG có trích nguồn.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính gemini-3.5-flash-lite đắt hơn gemini-3.1-flash-lite bao nhiêu lần cho
workload này? Nêu một trường hợp model chính xứng đáng với chi phí và một
trường hợp nên dùng model nhỏ:**

> **Khối lượng:** 30.000 lượt/ngày × 350 token = 10.500.000 token output/ngày
> = 10.500 đơn vị 1K.
>
> | Model | Giá output /1K | /ngày | /tháng |
> | --- | --- | --- | --- |
> | `gemini-3.5-flash-lite` | $0,0025 | $26,25 | $787,50 |
> | `gemini-3.1-flash-lite` | $0,0015 | $15,75 | $472,50 |
>
> **Đắt hơn 1,67 lần** (chênh ~$315/tháng, tính riêng output).
> - **Dùng model chính:** tác vụ mà một câu sai tốn hơn $315/tháng — tóm tắt
>   hợp đồng, sinh code, suy luận nhiều bước.
> - **Dùng model nhỏ:** tra cứu/phân loại đơn giản — như câu "bao nhiêu tỉnh
>   thành" ở Checkpoint 1, cả hai đều đúng mà model nhỏ nhanh gấp 2,2 lần.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Độ dài gần bằng nhau (309 vs 290 từ) vì cả hai đều chạm trần `max_tokens`,
> nhưng *từ vựng* và *ví dụ* khác hẳn. Bản cho trẻ 8 tuổi dùng phép so sánh đời
> thường (cuốn sổ chung, bạn An chép điều ước) và câu ngắn; bản chuyên gia dùng
> thuật ngữ (distributed ledger, SHA-256, Merkle Root, tấn công 51%) và cấu
> trúc phân mục. Cùng một câu hỏi, system prompt quyết định *cách* model trình
> bày — vai, giọng, độ sâu, ví dụ — chứ không đổi nội dung cốt lõi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> **Chênh lệch:** đoạn ~125 từ tiếng Việt cho `count_tokens` = 154, còn `từ/0.75`
> = 167 → ước lượng cao hơn **~8%** (thực tế 1,23 token/từ so với giả định 1,33).
>
> **Vì sao tiếng Việt tốn nhiều token hơn:** BPE học chủ yếu trên dữ liệu tiếng
> Anh, nên từ tiếng Anh phổ biến thường gộp thành 1 token, còn tiếng Việt thì
> không. Chữ có dấu (thanh, mũ, móc) bị mã hóa thành nhiều byte UTF-8 và hay bị
> tách nhỏ; thêm nữa tiếng Việt viết rời từng âm tiết ("khu vực", "văn hóa"),
> mỗi âm tiết dễ thành 1–2 token. Đo được ở đây: 1,23 token/từ (VI) so với 1,16
> (EN). Chênh không lớn vì o200k_base đã hỗ trợ Unicode khá tốt.
>
> *Lưu ý:* Gemini không có bảng mã trong tiktoken nên `count_tokens` dùng
> `o200k_base` (tokenizer GPT-4o) làm xấp xỉ — không phải token thật Gemini tính.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi dài và có người ngồi chờ trực tiếp —
> chatbot, trợ lý code, hỏi đáp: đo được ở đây, chữ đầu hiện sau 1,36s trong
> khi câu trả lời đầy đủ mất 2,22s, nên người dùng thấy phản hồi sớm hơn ~0,9s
> và không tưởng app bị treo. Nó cải thiện *cảm nhận* độ trễ chứ không giảm tổng
> thời gian. Non-streaming phù hợp hơn khi không có người xem trực tiếp hoặc cần
> nguyên vẹn kết quả trước khi dùng: gọi API trong backend, cần parse JSON/chạy
> tool trên toàn bộ output, hoặc phải kiểm duyệt nội dung trước khi hiển thị —
> khi đó xử lý từng mảnh vừa phức tạp vừa vô ích.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Delay cố định làm mọi client retry cùng một nhịp, nên nếu API quá tải thì cứ
> mỗi 1 giây nó lại nhận nguyên một đợt request dồn về — đúng lúc đang yếu lại
> bị đập thêm, khó hồi phục (thundering herd). Exponential backoff giãn khoảng
> chờ tăng dần (0,10 → 0,20 → 0,40s, đo được ở bài) nên tải retry giảm nhanh
> theo thời gian, cho server khoảng thở. Nếu hàng nghìn client cùng delay cố
> định giống nhau, chúng sẽ retry đồng bộ thành từng đợt nhọn lặp lại, có thể
> giữ server sập mãi. Thực tế còn thêm *jitter* (ngẫu nhiên hóa delay) để phá vỡ
> sự đồng bộ đó — backoff riêng lẻ vẫn có thể trùng pha giữa các client.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: trợ giảng thân thiện của khóa AI. System prompt:
> `"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."`
>
> Hai lựa chọn quan trọng:
> - **"ngắn gọn"** — chặn model viết lan man: vừa giảm token output (đỡ tốn phí,
>   phản hồi nhanh hơn) vừa hợp bối cảnh CLI, nơi câu trả lời dài rất khó đọc.
> - **"bằng tiếng Việt"** — ghim ngôn ngữ đầu ra. Không có dòng này, model dễ
>   chèn thuật ngữ hoặc cả câu tiếng Anh khi người dùng hỏi về AI; chỉ định rõ
>   giúp giọng nhất quán.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> **Hạn chế lớn nhất:** history bị cắt cứng còn 3 lượt (`history[-6:]`) — hỏi
> lại điều đã nói 4 lượt trước là trợ lý "quên" hẳn, vì message cũ bị xóa chứ
> không được tóm tắt.
>
> **Cải thiện — rolling summary:** thay vì bỏ message cũ, gộp chúng thành một
> đoạn tóm tắt ngắn. Cụ thể: khi history vượt 6 message, lấy các message sắp bị
> cắt gửi cho model kèm prompt "tóm tắt hội thoại trên trong 2-3 câu", rồi chèn
> bản tóm tắt vào đầu history dưới dạng một system message ("Bối cảnh trước
> đó: ..."). Nhờ vậy giữ được ngữ cảnh dài mà số token vẫn có trần, không phình
> vô hạn.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
