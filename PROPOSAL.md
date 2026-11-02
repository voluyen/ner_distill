# Proposal: Chưng cất tri thức cho NER sinh (text → JSON) trên MedGemma

## 1. Bối cảnh và vấn đề
NER y sinh hiện được giải quyết tốt bằng LLM lớn (MedGemma-27B), nhưng chi phí suy luận cao và khó triển khai. Mô hình nhỏ (MedGemma-4B) rẻ hơn khoảng 7 lần về tham số nhưng kém hơn khi chạy zero-shot. Cần một cách chuyển năng lực của teacher sang student mà **không cần fine-tune teacher** và vẫn giữ định dạng đầu ra có cấu trúc (JSON hợp lệ).

## 2. Mục tiêu
1. So sánh các phương pháp chưng cất logit và căn chỉnh biểu diễn trung gian cho NER dạng sinh: **DistiLLM (SKL, SRKL)**, **AMiD**, **MTA** (span alignment của hidden states), và các tổ hợp.
2. Đo xem student 4B đạt bao nhiêu phần hiệu năng của teacher 27B (few-shot) trên En-ViMedNER.
3. Kiểm tra MTA có bổ sung giá trị khi kết hợp với loss trên logit hay không (ở CoNLL-2003 với BERT, MTA tăng dev F1 nhưng không tăng test F1).

## 3. Bài toán và dữ liệu
- **Input**: câu văn bản y sinh. **Output**: danh sách JSON `[{"entity": "...", "type": "..."}]`, sắp theo vị trí, `[]` nếu không có thực thể.
- **Dữ liệu**: phần tiếng Anh của En-ViMedNER (dẫn xuất từ MedMentions ST21pv, 21 kiểu UMLS): 27,013 / 8,932 / 8,947 câu train / dev / test.
- **Chỉ số**: micro P/R/F1 trên cặp (văn bản thực thể, kiểu) theo từng câu; ngoài ra `span_f1` (bỏ qua kiểu) và `valid_json` (% đầu ra là JSON hợp lệ).

## 4. Phương pháp
| | Mô hình | Vai trò |
|---|---|---|
| Teacher | `google/medgemma-27b-it` | Không SFT, dùng few-shot (5 ví dụ cố định) |
| Student | `google/medgemma-4b-it` | LoRA r=64 rồi merge, chỉ huấn luyện bằng chưng cất |

- **Căn chỉnh token**: teacher thấy `prompt few-shot + JSON đích`, student thấy `prompt zero-shot + cùng JSON đích`. Hai bên cùng vị trí giám sát nên so sánh logit từng token. Loss chỉ tính trên token JSON và EOS.
- **Loss tổng**: `(1-r)·CE(gold) + r·KD`, với KD ∈ {SKL, SRKL, AMiD}.
- **MTA**: căn chỉnh hidden state theo span (từng từ và cụm danh/động từ từ spaCy) giữa các cặp lớp teacher:student `31:17` (từ) và `52:28`, `62:34` (cụm); projector lr 5e-4.
- Chỉ dùng các độ đo phân kỳ trên đầu ra gold, chưa dùng sinh thích ứng từ student (adaptive off-policy) của DistiLLM.

## 5. Thiết kế thực nghiệm
| Nhóm | Run | Mục đích |
|---|---|---|
| Cận | `teacher-eval`, `student-eval` | Cận trên (teacher few-shot) và cận dưới (student chưa huấn luyện) |
| Logit KD | `skl`, `srkl`, `amid` | So sánh các loss chưng cất |
| Biểu diễn | `mta_only` | Đóng góp riêng của MTA |
| Kết hợp | `srkl_mta`, `amid_mta` | Hiệu quả cộng gộp |

Lệnh chạy: `bash run_medgemma.sh` (tự kiểm tra truy cập HF, chuẩn bị dữ liệu, smoke test, rồi chạy toàn bộ và ghi `results_llm/RESULTS.md`). Chọn epoch tốt nhất theo F1 giải mã tham lam trên 500 câu dev, rồi báo cáo trên toàn bộ dev và test.

## 6. Giả thuyết
- H1: Student sau KD vượt rõ student zero-shot và tiến gần teacher few-shot.
- H2: SRKL/SKL ổn định hơn FKL thuần nhờ tránh quá khớp mode/bao phủ.
- H3: MTA giúp ở dev (hội tụ, valid_json) nhưng mức tăng test có thể nhỏ, cần nhiều seed mới kết luận.

## 7. Tài nguyên và rủi ro
- GPU: khoảng 70–90 GB (teacher bf16 ~54 GB + student LoRA); batch 4 × grad-accum 8 do tensor logit (token × 262k vocab) lớn. Chạy được trên H200 143 GB khi tiến trình khác giải phóng bộ nhớ.
- Rủi ro: (a) mô hình gated trên HF, cần tài khoản đã chấp nhận licence; (b) giới hạn tốc độ HF khi tải dữ liệu (đã có cơ chế thử lại); (c) một số phần chưa kiểm chứng trên Gemma 3 thật (nạp `Gemma3ForConditionalGeneration` dạng text-only, workaround `token_type_ids`, regex LoRA) nên cần smoke test trước; (d) chênh lệch giữa các phương pháp có thể nhỏ hơn nhiễu một seed.

## 8. Kế hoạch
1. Hoàn tất tải dữ liệu và smoke test (`--train-num 50`).
2. Chạy `teacher-eval`, `student-eval`, rồi 6 run KD (seed 42).
3. Với 2–3 cấu hình tốt nhất, chạy thêm các seed để có trung bình ± độ lệch chuẩn.
4. Phân tích lỗi trên `test_predictions.jsonl` (JSON hỏng, lỗi theo từng kiểu thực thể).

## 9. Kết quả kỳ vọng
Bảng F1 (dev/test), `span_f1`, `valid_json`, thời gian huấn luyện cho từng phương pháp; kết luận về việc loss nào và có hay không MTA đáng dùng cho NER sinh trong miền y sinh. *Hiện chưa có kết quả huấn luyện nào cho pipeline MedGemma; các mục trên là kế hoạch.*

## 10. Tham chiếu
DistiLLM (Ko et al., 2024), AMiD, MTA; En-ViMedNER (`nhuvo/En-ViMedNER`); MedGemma (Google). Mã trong repo: `nerllm/`, `nerkd/losses.py`, `scripts/run_llm.sh`.
