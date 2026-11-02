# Báo cáo thực nghiệm: chưng cất MedGemma-27B → MedGemma-4B cho NER sinh (text → JSON)

Dữ liệu chạy: 21–23/10/2026 trên 1 GPU H200 (143 GB), seed 42. Kết quả gốc: `results_llm/RESULTS.md`, `results_llm/*/*/results.json`, log ở `logs/llm/`.

## 1. Tóm tắt
- Bài toán: nhập một câu y sinh, sinh ra danh sách JSON `[{"entity", "type"}]` gồm 21 kiểu UMLS (En-ViMedNER).
- Student MedGemma-4B được huấn luyện bằng LoRA từ teacher MedGemma-27B (không SFT, few-shot) với các loss AMiD, MTA và DistiLLM (SKL/SRKL).
- **Tốt nhất: MTA only, test F1 = 64,29** (dev 64,93). AMiD + MTA đạt 61,82 và AMiD đạt 60,46.
- **Ba run DistiLLM (SKL, SRKL, SRKL+MTA) bị hỏng, F1 = 0**, do loss thành NaN. Đó là lỗi số học, không phải kết quả của phương pháp.
- **Điểm của teacher (23,5) bị đánh giá thấp so với năng lực thật** vì prompt không nêu 21 kiểu nhãn; vì vậy kết quả hiện tại **chưa chứng minh chưng cất có lợi**: phần lớn điểm của student đến từ CE trên nhãn gold.
- Chỉ 1 seed, chưa có baseline SFT thuần; các chênh lệch vài điểm F1 chưa đủ để kết luận.

## 2. Thiết lập thực nghiệm

### 2.1 Dữ liệu
Phần tiếng Anh của `nhuvo/En-ViMedNER` (dẫn xuất từ MedMentions ST21pv): train 27.013 câu (122.017 thực thể), dev 8.932 (40.844), test 8.947 (40.088). Khi huấn luyện loại 4 câu dài hơn 768 token (còn 27.009). Dev/test không bị loại câu nào.

Prompt (chat template của Gemma), phần trả lời kết thúc bằng `<end_of_turn>`:
```
Extract all biomedical named entities from the text. Answer with a JSON list of objects with the keys "entity" (the exact span from the text) and "type".

Text: <câu>
Entities:
```
Đầu ra là JSON sắp theo vị trí, `[]` nếu không có thực thể. Prompt **không liệt kê 21 kiểu** (`types_in_prompt=false`).

### 2.2 Mô hình
| Vai trò | Mô hình | Cách dùng |
|---|---|---|
| Teacher | `google/medgemma-27b-it` | Không SFT, prompt few-shot 5 ví dụ cố định |
| Student | `google/medgemma-4b-it` | LoRA r=64, α=128, dropout 0,05; merge vào checkpoint khi kết thúc |

Hai mô hình dùng chung tokenizer Gemma 3 (vocab 262.144). Teacher thấy `prompt few-shot + JSON đích`, student thấy `prompt zero-shot + cùng JSON đích`, nên các vị trí giám sát trùng nhau và logit được so từng token. Loss chỉ tính trên token JSON và EOS.

### 2.3 Loss và siêu tham số
- Loss tổng: `(1−r)·CE(gold) + r·KD`, với `kd_ratio r = 0,5`.
- KD ∈ {SKL, SRKL (DistiLLM, skew α=0,1), AMiD}; `mta_only` dùng `kd-type none`, tức CE trên gold cộng loss MTA (teacher chỉ đóng góp qua hidden state).
- MTA: căn chỉnh hidden state theo span (từng từ và cụm danh/động từ từ spaCy), cặp lớp teacher:student `31:17` (từ), `52:28` và `62:34` (cụm), trọng số 2,0.
- Tối ưu: lr 1e-4, warmup 3%, cosine, 3 epoch, batch 4 × tích lũy 8 (hiệu dụng 32), clip grad 1,0, bf16, gradient checkpointing, attention `eager`.
- Mỗi epoch: dev loss trên toàn bộ dev và F1 giải mã tham lam trên 500 câu dev đầu tiên (tối đa 512 token sinh). Giữ epoch có dev F1 tốt nhất, rồi chấm trên **toàn bộ dev và test**.
- Môi trường: `.venv-gemma` (torch 2.6.0, transformers 4.57.6, peft 0.21.2, datasets 3.6.0, pyarrow 18.1.0). Mỗi run KD mất khoảng 4,5–5,5 giờ cho 3 epoch (mỗi epoch ~1,5 giờ).

### 2.4 Chỉ số
Micro P/R/F1 trên cặp (chuỗi thực thể, kiểu) theo từng câu, so khớp chính xác dạng multiset. Ngoài ra: `span_f1` (bỏ qua kiểu) và `valid_json` (% đầu ra là một danh sách JSON hợp lệ). JSON hỏng được cứu từng object. Chi tiết trong `nerllm/metrics.py`.

### 2.5 Các run
`teacher-eval`, `student-eval` (5-shot, không huấn luyện), `skl`, `srkl`, `amid`, `mta_only`, `srkl_mta`, `amid_mta`.

## 3. Kết quả

### 3.1 Bảng chính (toàn bộ dev/test)
| Phương pháp | Dev F1 | Test P | Test R | **Test F1** | Span F1 | Valid JSON % |
|---|---|---|---|---|---|---|
| Teacher 27B (5-shot, không huấn luyện) | 24,58 | 24,78 | 22,36 | 23,51 | 54,23 | 0,67 |
| Student 4B (5-shot, không huấn luyện) | 16,59 | 16,62 | 14,56 | 15,52 | 46,28 | 5,17 |
| SKL (DistiLLM) | 0 | 0 | 0 | 0 | 0 | 0 |
| SRKL (DistiLLM) | 0 | 0 | 0 | 0 | 0 | 0 |
| AMiD | 60,86 | 58,55 | 62,51 | 60,46 | 68,89 | 99,98 |
| **MTA only** | **64,93** | 63,57 | 65,02 | **64,29** | 71,77 | 99,97 |
| SRKL + MTA | 0 | 0 | 0 | 0 | 0 | 0 |
| AMiD + MTA | 62,32 | 60,56 | 63,12 | 61,82 | 69,79 | 99,97 |

### 3.2 Dev F1 (500 câu) theo epoch
| Run | Epoch 1 | Epoch 2 | Epoch 3 | Epoch chọn |
|---|---|---|---|---|
| AMiD | **63,42** | 63,10 | 62,54 | 1 |
| MTA only | 63,91 | **65,40** | 65,23 | 2 |
| AMiD + MTA | 60,74 | 62,56 | **63,52** | 3 |

## 4. Phân tích

1. **MTA only cao nhất** (64,29), hơn AMiD 3,8 điểm và AMiD + MTA 2,5 điểm ở test. Chạy này không có loss logit từ teacher, nên phần lớn cải thiện là từ CE trên gold; MTA cho thêm tín hiệu biểu diễn. Do thiếu baseline SFT thuần (`kd-ratio` 0) và chỉ có 1 seed, chưa tách được đóng góp của MTA.
2. **AMiD + MTA > AMiD** (+1,4 test F1), đáng chú ý nhưng nằm trong mức dao động có thể có giữa các seed.
3. **Điểm teacher thấp** (23,5 F1, `valid_json` 0,67%). Đọc `test_predictions.jsonl` thấy:
   - Teacher bọc đầu ra trong rào markdown ` ```json ` ở 99,3% câu. Bộ kiểm tra `valid_json` coi đây là không hợp lệ, nhưng F1 vẫn đọc được từng object nên không bị trừ điểm vì việc này.
   - **34% thực thể dự đoán có nhãn nằm ngoài 21 kiểu** (`Disease or Syndrome` 2.922 lần, `Gene or Genome` 1.594, `Protein` 1.013, `Laboratory Procedure` 643...), vì prompt không liệt kê kiểu. Hệ quả: Span F1 là 54,2 nhưng F1 chỉ 23,5.
   - Quy ước gán nhãn của MedMentions (ví dụ gán "present" là Finding) khó học qua 5 ví dụ.
   - Vì vậy student sau KD (60–64) vượt teacher chủ yếu nhờ học trực tiếp từ nhãn gold, không phải nhờ vượt trội về tri thức.
4. **Run DistiLLM thất bại.** `srkl` có loss NaN từ khoảng bước 60, `srkl_mta` ngay từ khoảng bước 20; `skl` không NaN ở log train nhưng dev loss NaN từ epoch 2 (tại epoch 1 đã lưu checkpoint F1 = 0). Nghi do `log` hoặc `exp` trên xác suất bằng 0 khi trộn phân phối với vocab 262k ở bf16; chưa kiểm tra mã nguồn nên chưa chắc nguyên nhân. Ba run này tốn khoảng 5–8 giờ GPU mỗi run mà không có kết quả.
5. **Số epoch**: AMiD tốt nhất ở epoch 1 (dev F1 giảm dần), MTA only ở epoch 2; 3 epoch có phần thừa với CE+LoRA.

## 5. Hạn chế
- Một seed (42); chưa có độ lệch chuẩn.
- Không có baseline student SFT thuần; không tách được lợi ích của KD so với CE trên gold.
- Teacher bị prompt kém (không nêu kiểu nhãn), nên không phải một teacher mạnh đúng nghĩa.
- Kiểm tra dev trong lúc huấn luyện chỉ dùng 500 câu.
- Chấm theo cặp (chuỗi, kiểu) khớp chính xác, không có vị trí ký tự và không có điểm khớp lỏng.
- Lỗi NaN chưa được chẩn đoán đến cùng.

## 6. Việc đề xuất tiếp theo
1. Sửa độ ổn định số học của `skl/srkl` trong `nerkd/losses.py` (tính trong float32, clamp log-prob), kiểm tra nhanh vài trăm câu rồi chạy lại 3 run lỗi.
2. Chạy lại `teacher-eval` với `--types-in-prompt` (thử trên ~500 câu dev trước) để có teacher đúng nghĩa.
3. Thêm baseline student SFT thuần (`kd-ratio` 0) và chạy nhiều seed cho các cấu hình tốt nhất.
4. Dừng sớm ở epoch 1–2 để giảm chi phí.
5. Bổ sung F1 theo vị trí ký tự và khớp lỏng vào `metrics.py`, rồi chấm lại từ `test_predictions.jsonl` có sẵn.

## 7. Tái lập
```bash
bash run_medgemma.sh                         # toàn bộ (cần ~90 GB GPU, HF đã được cấp quyền 2 mô hình)
EXPS="amid mta_only" bash run_medgemma.sh    # tập con
```
Tải dữ liệu bị giới hạn tốc độ HF: dùng `scripts/download_envimed.sh` (tự nghỉ và thử lại khi gặp lỗi 429).
