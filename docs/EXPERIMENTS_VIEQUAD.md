# Đối chứng truy xuất VieQuADRetrieval (25/09/2026)

Chạy tái lập ngày 27/09/2026: `runs/20260927T155958Z-viequad/results.json`; điểm test và CI khớp lần chạy gốc. Đây là tái lập cùng giao thức, không phải một test set mới hay bằng chứng cải tiến mới.

Kết quả gốc: `runs/20260925T104214Z-viequad/results.json` và `per_query.jsonl`. Dữ liệu từ [mteb/VieQuADRetrieval](https://huggingface.co/datasets/mteb/VieQuADRetrieval), revision `f956535`: 2.490 đoạn, 2.048 câu hỏi, 4.096 qrels. Đây là đoạn Wikipedia tiếng Việt, **không phải** PDF học liệu hoặc kiểm tra OCR/chunking.

## Giao thức cố định

- Chỉ mục Qdrant: `data/indexes/viequad_f956535`, BGE-M3 1.024 chiều, model snapshot `5617a9f61b028005a4858fdac845db406aefb181`; BM25 tách từ bằng khoảng trắng. Mỗi nhánh lấy tối đa 100 ứng viên trên cùng corpus. Không dùng reranker hay LLM trong phép đo này.
- Bỏ 33 câu có qrels ở nhiều tiêu đề bài nguồn. Chia theo tiêu đề bài nguồn bằng seed 42: 100 câu dev từ 6 bài, 200 câu test từ 13 bài còn lại; không trùng bài nguồn giữa dev/test. Danh sách query ID và bài nguồn nằm trong `results.json`.
- Chọn α của `α·BM25 + (1−α)·dense` **chỉ trên dev**, lưới `{0; 0,25; 0,5; 0,75; 1}` theo nDCG@10. Dev chọn α=0,25 (nDCG@10 0,655); dense 0,636, RRF 0,608. Sau đó đánh giá test một lần. Các baseline BM25, dense, RRF dùng cùng tập ứng viên; adaptive α=0,5/β=0,3 là biến thể đối chiếu, không được chọn từ test.

| Phương pháp | nDCG@10 test | MRR@10 | Hit@1 | Hit@10 |
|---|---:|---:|---:|---:|
| BM25 | 0,471 | 0,637 | 0,505 | 0,875 |
| Dense | 0,574 | 0,742 | 0,615 | 0,950 |
| RRF | 0,574 | 0,799 | 0,695 | 0,965 |
| Adaptive (chưa tune) | 0,538 | 0,733 | 0,600 | 0,945 |
| Weighted α=0,25 (chọn trên dev) | **0,606** | **0,812** | **0,705** | **0,970** |

Chênh nDCG@10 của α=0,25 so với dense là +0,0317; CI bootstrap ghép cặp 95% theo **bài nguồn** là [0,0153; 0,0457]. So với RRF: +0,0313, CI [0,0152; 0,0476]. Ở top-1, cấu hình đã chọn sửa đúng 22 câu mà dense sai, nhưng làm sai 4 câu dense đúng; 119 câu cả hai đúng và 55 câu cả hai sai. Một ca thua là câu về độ mặn biển Caspi (query `2109`): dense đặt một đoạn đúng đầu tiên, weighted đưa đoạn khác lên trước. Các trường hợp cụ thể có thể kiểm tra trong `per_query.jsonl`.

## Diễn giải đúng phạm vi

Kết quả ủng hộ **tuning trọng số** trên bộ đối chứng tiếng Việt này, nhưng không chứng minh cấu hình tối ưu cho tài liệu học tập. CI chỉ dựa trên 13 nhóm bài test, không thay cho nhiều miền độc lập. Qrels công khai đánh dấu hai đoạn/câu nhưng có thể bỏ sót đoạn khác cũng trả lời được. Pilot CSDL cũ đã được dùng để debug nên không phải held-out test và đã được dọn khỏi repo; không gộp điểm của nó với VieQuAD. Cần benchmark học liệu có nguồn/bằng chứng và nhãn người duyệt trước khi tuyên bố đạt mục tiêu đa ngành/đa cấp.

Chạy lại: `conda run -n ppl-rag python run_viequad_benchmark.py` sau khi tải dữ liệu đúng revision theo `README.md` và khởi động Qdrant.

## Tuning adaptive tiếp theo (27/09/2026, thăm dò)

Runner chung đã được thử trên dữ liệu/qrels công khai xuất nguyên nhãn từ VieQuAD; không dùng nhãn tự sinh. Audit dev: `runs/tuning-20260927T161219191541Z/dev_selection.json`; cấu hình khóa: `selected_config.yaml` cùng thư mục. Quét 25 cặp α₀∈{0;0,25;0,5;0,75;1}, β∈{−0,5;−0,25;0;0,25;0,5}, chọn theo nDCG@10 của 100 câu dev. Chọn α₀=0, β=0,25, dev 0,6592. Hàm adaptive vẫn giới hạn trọng số cuối trong [0,1;0,9], nên α₀=0 **không** có nghĩa chỉ dùng dense.

Kết quả runner: `runs/20260927T161235Z-4eb378de/metrics.json`, ranking đầy đủ từng câu trong `per_query.csv`, hash đầu vào và config được đóng băng. Trên 200 câu test cũ: E6 adaptive đã tune nDCG@10=0,6048, so với adaptive cũ 0,5381; dense 0,5739; hybrid cố định đã tune 0,6056. E6−dense=+0,03095, CI bootstrap theo 13 bài nguồn [0,01196;0,04751]. **Adaptive đã tune không vượt hybrid cố định đã tune.** CI trong `paired_comparisons.json` là so sánh thăm dò, chưa hiệu chỉnh nhiều phép thử.

Đây là phân tích tiếp trên tập test đã được xem trước đó, không phải kiểm định xác nhận trên một test set mới. Không dùng nó để tuyên bố tính mới vượt trội hay accuracy học liệu. Benchmark học liệu cần câu hỏi độc lập, pooling/chấm nhãn người duyệt và test chưa được sử dụng để phát triển hệ thống.
