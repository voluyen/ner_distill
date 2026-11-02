# Chưng cất NER theo protocol IWD: Qwen2.5-7B → Qwen2.5-0.5B

Ngày chạy: 29/09/2026. Kết quả gốc: `results_iwd/qwen/`, bảng tổng hợp: `results_iwd/RESULTS.md`.

## Tóm tắt

- **Teacher Qwen2.5-7B (LoRA) chỉ đạt 85.83 F1 trên CoNLL test**, thấp hơn nhiều so với bert-base-cased (91.14). Nguyên nhân chính là Qwen là model causal, mỗi token chỉ nhìn được ngữ cảnh bên trái. Khoảng cách giữa teacher và student là 2.26 F1 (CoNLL), 3.87 (WNUT-17) và 2.91 (OntoNotes).
- **In-domain (CoNLL test): AMiD+MTA tốt nhất**, đạt 84.35 ± 0.27. Mức này hơn student không KD 0.78 F1 (paired t-test p = 0.024, thắng 4/5 seed) và thu hẹp được 34% khoảng cách với teacher. Xếp sau là SRKL (DistiLLM) 84.14 ± 0.19 và SRKL+MTA 84.13 ± 0.38. Vanilla KD (FKL) chỉ tăng 0.32 và không có ý nghĩa thống kê.
- **MTA giúp AMiD rõ rệt**: AMiD đứng một mình còn thấp hơn baseline (−0.13), nhưng AMiD+MTA tăng +0.91 so với AMiD. Trên SRKL, MTA không đổi CoNLL nhưng tăng OntoNotes thêm +0.49.
- **OOD OntoNotes: hai biến thể có MTA dẫn đầu**. AMiD+MTA đạt 71.67 so với 70.99 của baseline, thắng cả 5/5 seed (p = 0.043).
- **OOD WNUT-17: mọi phương pháp KD đều thấp hơn baseline** (từ −0.75 đến −1.61 F1), dù teacher cao hơn student 3.9 F1. Student không thừa hưởng được độ bền OOD của teacher trên dữ liệu mạng xã hội. Đây đúng là câu hỏi mà IWD đặt ra ("Do Student LLMs Inherit OOD Robustness?"). Độ lệch chuẩn trên WNUT khoảng 2 F1, nên chỉ riêng mức giảm của AMiD là có ý nghĩa (p = 0.014, thua cả 5/5 seed).
- **Lưu ý thống kê**: bảng có 18 phép so sánh (6 phương pháp × 3 tập). Sau hiệu chỉnh Bonferroni (ngưỡng 0.0028), không phép so sánh nào còn có ý nghĩa. Các kết luận ở trên chỉ nên xem là xu hướng, và cần thêm seed để khẳng định.

## Setup

Protocol theo IWD (Kannangara & Dutta, arXiv 2609.22566): chỉ train trên CoNLL-2003, báo cáo F1 trên CoNLL test, rồi eval OOD trên **test** của WNUT-17 và OntoNotes 5.0. Mỗi ô là mean ± std qua 5 seed.

| Hạng mục | Cấu hình |
|---|---|
| Teacher | Qwen/Qwen2.5-7B, `AutoModelForTokenClassification`, LoRA r=16, α=32, dropout 0.05 trên mọi lớp linear. Head train đầy đủ, base để bf16, bật gradient checkpointing. lr 1e-4, chọn từ {1e-4, 2e-4} theo dev F1. Train 5 epoch, seed 42, sau đó merge adapter. 7.07B tham số (không tính lm_head) |
| Student | Qwen/Qwen2.5-0.5B, **full fine-tune**, 494M tham số. lr 1e-5, chọn từ {1e-5, 2e-5, 5e-5} theo dev F1 (seed 42) |
| Chung | 5 epoch, batch 32, AdamW (weight decay 0.01), warmup 10%, bf16 autocast, max length 256, ở mức câu (sentence-level) |
| KD | loss = 0.5·CE + 0.5·(KD + w·MTA), T = 1 (bằng τ = 1.0 của IWD). Teacher đóng băng, chạy bf16 |
| Các phương pháp | `fkl` (vanilla KD, trùng với baseline của IWD). DistiLLM: `skl`, `srkl` (skew α = 0.1). AMiD: `amid` (div ab, α_ab = 0.2, β_ab = 0.7, order pr, α = 0.5, λ = 0.5). MTA: `srkl_mta`, `amid_mta` (w = 2.0) |
| Layer cho MTA | word `14:12`, phrase `23:20 28:24`, ánh xạ 28 layer của teacher sang 24 layer của student theo cùng tỉ lệ với cặp BERT (`6:3 | 10:5 12:6`) |
| Tokenize | Tokenizer byte-level BPE của Qwen làm hỏng từ khi nhận từng từ riêng lẻ (" founded" thành "f"+"ounded"). Vì vậy tokenize nguyên câu, rồi map token về từ bằng char offset |
| Vị trí nhãn | Sub-token **cuối** của mỗi từ. Model là causal, nên đây là vị trí duy nhất đã nhìn thấy trọn cả từ |
| Chọn checkpoint | Epoch có CoNLL dev F1 cao nhất. Không dùng test hay dữ liệu OOD để chọn |
| Metric | Entity-level micro F1, khớp chính xác cả span lẫn loại, tách span kiểu conlleval (seqeval) |
| Seed | Student: 1–5, mỗi seed chạy đủ mọi phương pháp. Teacher: một lần duy nhất (seed 42) |

**Ánh xạ nhãn OOD.** IWD không công bố cách map, nên dưới đây là cách của chúng tôi (xem `nerkd/ood.py`):

| Tập | → PER | → ORG | → LOC | Bỏ qua | Coi là O |
|---|---|---|---|---|---|
| WNUT-17 | person | corporation, group | location | product, creative-work | — |
| OntoNotes 5.0 | PERSON | ORG | GPE, LOC | NORP, FAC, EVENT, WORK_OF_ART, LAW, LANGUAGE, PRODUCT | DATE, TIME, MONEY, PERCENT, QUANTITY, ORDINAL, CARDINAL |

"Bỏ qua" nghĩa là span gold bị loại khỏi phép chấm, cùng với mọi span dự đoán chồng lên nó. Dự đoán MISC luôn bị loại. Sau khi map, test WNUT-17 còn 810 entity gold và OntoNotes còn 6,202.

## Kết quả chính (entity F1, mean ± std, 5 seed)

| Phương pháp | CoNLL dev | CoNLL test | WNUT-17 | OntoNotes |
|---|---|---|---|---|
| Teacher Qwen2.5-7B (LoRA, 1 run) | 91.84 | 85.83 | 48.38 | 73.90 |
| Student 0.5B, không KD | 89.96 ± 0.25 | 83.57 ± 0.48 | **44.51 ± 2.07** | 70.99 ± 0.13 |
| Vanilla KD (FKL) | 90.12 ± 0.28 | 83.89 ± 0.29 | 43.66 ± 2.49 | 71.05 ± 0.34 |
| DistiLLM SKL | 90.00 ± 0.32 | 83.72 ± 0.28 | 43.68 ± 1.98 | 71.23 ± 0.90 |
| DistiLLM SRKL | 90.05 ± 0.29 | 84.14 ± 0.19 | 43.76 ± 2.30 | 71.07 ± 0.28 |
| AMiD | 90.08 ± 0.35 | 83.44 ± 0.40 | 42.91 ± 2.48 | 70.88 ± 0.76 |
| SRKL + MTA | **90.30 ± 0.20** | 84.13 ± 0.38 | 43.38 ± 1.75 | 71.56 ± 0.56 |
| AMiD + MTA | 90.22 ± 0.38 | **84.35 ± 0.27** | 43.20 ± 2.40 | **71.67 ± 0.55** |

## So với student không KD (paired t-test theo seed, n = 5)

Mỗi ô ghi: Δ F1 / p-value / số seed thắng. Dòng cuối là phần trăm khoảng cách teacher–student được thu hẹp trên CoNLL.

| Phương pháp | CoNLL test | WNUT-17 | OntoNotes | Thu hẹp gap (CoNLL) |
|---|---|---|---|---|
| FKL | +0.32 / 0.34 / 2 | −0.86 / 0.44 / 1 | +0.06 / 0.60 / 3 | 14% |
| SKL | +0.15 / 0.62 / 3 | −0.83 / 0.18 / 1 | +0.24 / 0.55 / 3 | 7% |
| SRKL | +0.57 / 0.10 / 4 | −0.75 / 0.11 / 1 | +0.08 / 0.57 / 4 | 25% |
| AMiD | −0.13 / 0.56 / 1 | **−1.61 / 0.014 / 0** | −0.11 / 0.72 / 2 | −6% |
| SRKL + MTA | +0.55 / 0.13 / 4 | −1.13 / 0.41 / 1 | +0.57 / 0.08 / 4 | 25% |
| AMiD + MTA | **+0.78 / 0.024 / 4** | −1.31 / 0.39 / 1 | **+0.68 / 0.043 / 5** | 34% |

So với vanilla KD (FKL), chỉ SRKL tốt hơn trên CoNLL (p = 0.021). AMiD+MTA cao hơn FKL 0.46 trên CoNLL (p = 0.11) và 0.62 trên OntoNotes (p = 0.056).

## Nhận xét

1. **Teacher causal yếu là điểm nghẽn.** Teacher 7B ở chế độ causal chỉ đạt 85.8 trên CoNLL test, thua cả DistilBERT (89.3). Nó cũng chênh 6 F1 giữa dev và test (91.8 so với 85.8), trong khi BERT chỉ chênh khoảng 4. Mỗi token chỉ thấy ngữ cảnh bên trái, nên nhãn của từ đầu một entity (ví dụ "New" trong "New York") được quyết định trước khi model thấy phần còn lại. Khoảng cách teacher–student vì thế chỉ còn 2.3 F1, nên dư địa để KD cải thiện cũng nhỏ.
2. **Chỉ các phương pháp có alignment hidden state (MTA) cải thiện đều được cả in-domain lẫn OntoNotes.** Các loss chỉ dựa trên logit đạt nhiều nhất +0.57 trên CoNLL và gần như không đổi trên OntoNotes. Trên nhãn BIO (9 lớp), phân phối của teacher gần như one-hot, nên các divergence khác nhau không tách biệt được nhiều.
3. **WNUT-17 đi ngược xu hướng.** KD làm giảm chủ yếu recall trên tweet: recall của student không KD là 56.3, còn các phương pháp KD nằm trong khoảng 53.6–55.2, trong khi precision chỉ giảm 0.5–1 điểm. Teacher 48.4 so với student 44.5 cho thấy độ bền OOD nằm trong teacher mà không truyền qua logit KD. Phương sai giữa các seed trên WNUT (std khoảng 2) lớn gấp bốn lần trên CoNLL, nên cần khoảng 10 seed hoặc bootstrap để kết luận chắc chắn.
4. **Không so trực tiếp được với con số của IWD.** IWD dùng các cặp khác (DeBERTa-v3-base → xsmall, và Qwen2.5-1.5B → 0.5B), không công bố cách map nhãn, và teacher 7B của chúng tôi lớn hơn. Chỉ nên so xu hướng, ví dụ vanilla KD có giúp OOD hay không.

## Hạn chế

- Hyperparameter của AMiD (α, λ, α_ab, β_ab) và trọng số MTA được giữ nguyên từ cặp BERT, chưa tune lại cho Qwen.
- Teacher chỉ có 1 seed. Teacher train bằng LoRA vì lúc bắt đầu GPU chỉ còn khoảng 67GB trống (một tiến trình vLLM khác đang chiếm phần còn lại). Full fine-tune 7B với AdamW fp32 cần khoảng 120GB.
- Metric là F1 mức câu, không dùng ngữ cảnh mức văn bản.
- Các run BERT và DeBERTa theo cùng protocol đã bị dừng giữa chừng khi chuyển sang Qwen (BERT có 2 seed, DeBERTa có 1 seed). Kết quả dở dang vẫn nằm trong `results_iwd/bert` và `results_iwd/deberta` nhưng không được dùng trong report này.

## Hướng tiếp theo

1. **Bỏ causal mask** (dùng attention hai chiều như LS-unLLaMA) cho cả teacher lẫn student. Đây có lẽ là thay đổi tăng F1 nhiều nhất cho cả hai.
2. Tune AMiD và trọng số MTA trên dev cho cặp Qwen.
3. Chạy thêm seed, hoặc dùng bootstrap test trên dự đoán, cho WNUT-17.
4. Chạy xong cặp BERT và DeBERTa để có bảng so sánh giữa encoder và decoder.

## Chạy lại

```bash
bash scripts/run_iwd.sh                       # teacher (nếu chưa có) + 7 cấu hình x 5 seed, JOBS=2 mặc định
.venv/bin/python -m nerkd.results_iwd results_iwd
```
Teacher trong bảng được chọn từ sweep lr (`results_iwd/qwen_sweep/`), rồi chuyển vào `results_iwd/qwen/sft/Qwen_Qwen2.5-7B`. Với 3 run song song trên H200, mỗi run KD mất khoảng 13 phút, mỗi run SFT khoảng 7 phút, và mỗi teacher LoRA khoảng 18 phút.
