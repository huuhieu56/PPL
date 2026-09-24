# NGHIÊN CỨU CẢI THIỆN ĐỘ CHÍNH XÁC TRUY XUẤT TÀI LIỆU HỌC TẬP BẰNG PHƯƠNG PHÁP HYBRID RETRIEVAL TRONG HỆ THỐNG RAG

## MỤC LỤC
* MỤC LỤC
* DANH MỤC CÁC KÝ HIỆU, CÁC CHỮ VIẾT TẮT
* DANH MỤC CÁC BẢNG
* DANH MỤC CÁC HÌNH
* CHƯƠNG 1: CƠ SỞ LÝ LUẬN VÀ TỔNG QUAN NGHIÊN CỨU
  * 1.1. Các khái niệm liên quan đến truy xuất tài liệu trong hệ thống RAG
    * 1.1.1. Khái niệm mô hình RAG (Retrieval-Augmented Generation)
    * 1.1.2. Khái niệm truy xuất thông tin và các chỉ số đánh giá
    * 1.1.3. Khái niệm truy xuất từ vựng (Sparse Retrieval)
    * 1.1.4. Khái niệm truy xuất ngữ nghĩa (Dense Retrieval)
    * 1.1.5. Khái niệm truy xuất hỗn hợp (Hybrid Retrieval) và tái xếp hạng (Reranking)
    * 1.1.6. Khái niệm và đặc thù của dữ liệu tài liệu học tập
  * 1.2. Tổng quan nghiên cứu trong nước và ngoài nước
    * 1.2.1. Các nghiên cứu trên thế giới
    * 1.2.2. Các nghiên cứu tại Việt Nam
    * 1.2.3. Nhận xét và khoảng trống nghiên cứu (Research Gap)
  * 1.3. Định hướng nghiên cứu của đề tài
    * 1.3.1. Mục tiêu nghiên cứu
    * 1.3.2. Nhiệm vụ nghiên cứu
    * 1.3.3. Đối tượng và phạm vi nghiên cứu
  * 1.4. Kết luận chương
* CHƯƠNG 2: PHÂN TÍCH, THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG TRUY XUẤT TÀI LIỆU HỌC TẬP BẰNG HYBRID RETRIEVAL
  * 2.1. Phân tích yêu cầu bài toán nghiên cứu
    * 2.1.1. Phân tích đối tượng sử dụng và nhu cầu truy xuất
    * 2.1.2. Phân tích bối cảnh và giới hạn của hệ thống
    * 2.1.3. Câu hỏi nghiên cứu và giả thuyết kiểm định
    * 2.1.4. Yêu cầu chức năng và phi chức năng
  * 2.2. Phân tích bài toán truy xuất tài liệu học tập
    * 2.2.1. Mô hình hóa bài toán
    * 2.2.2. Đặc thù dữ liệu và phân nhóm truy vấn tiếng Việt học thuật
  * 2.3. Thiết kế tổng thể hệ thống
    * 2.3.1. Nguyên tắc thiết kế
    * 2.3.2. Kiến trúc tổng thể và luồng xử lý dữ liệu
    * 2.3.3. Thiết kế luồng xử lý dữ liệu chi tiết
    * 2.3.4. Thiết kế giao tiếp giữa các thành phần
  * 2.4. Xây dựng và xử lý bộ dữ liệu thực nghiệm
    * 2.4.1. Thu thập và chuẩn hóa tài liệu
    * 2.4.2. Phân đoạn tài liệu (Structure-aware Chunking)
    * 2.4.3. Xây dựng tập benchmark và kiểm soát rủi ro sai lệch
  * 2.5. Xây dựng các phương pháp truy xuất và Hybrid Retrieval
    * 2.5.1. Cấu hình đối chứng Sparse Retrieval (BM25)
    * 2.5.2. Cấu hình đối chứng Dense Retrieval
    * 2.5.3. Cấu hình Hybrid Retrieval và phương pháp dung hợp
    * 2.5.4. Cấu hình tái xếp hạng (Cross-Encoder Reranking)
    * 2.5.5. Ma trận cấu hình thực nghiệm và kiểm định loại trừ (Ablation Study)
  * 2.6. Thiết kế thực nghiệm và phương pháp đánh giá
    * 2.6.1. Quy trình thực nghiệm chuẩn tắc
    * 2.6.2. Các chỉ số đánh giá định lượng
  * 2.7. Triển khai và tích hợp hệ thống RAG
    * 2.7.1. Kiến trúc triển khai tổng thể
    * 2.7.2. Triển khai module lập chỉ mục (Indexing Module)
    * 2.7.3. Triển khai dịch vụ Retrieval (Retrieval Service)
    * 2.7.4. Tích hợp với pipeline RAG
    * 2.7.5. Tối ưu hiệu năng
    * 2.7.6. Khả năng mở rộng và bảo trì
  * 2.8. Kết luận chương
* CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ
  * 3.1. Môi trường và thiết lập thực nghiệm
  * 3.2. Bộ dữ liệu và benchmark
  * 3.3. Tinh chỉnh siêu tham số trên tập dev
  * 3.4. Kết quả trên tập test
    * 3.4.1. RQ1 – So sánh BM25 và Dense Retrieval theo nhóm truy vấn
    * 3.4.2. RQ2 – Hiệu quả của Hybrid Retrieval
    * 3.4.3. RQ3 – Đóng góp và chi phí của Reranker
    * 3.4.4. Phân tích độ nhạy theo nguồn câu hỏi
  * 3.5. Phân tích lỗi
  * 3.6. Các hướng khai thác mở rộng
  * 3.7. Thảo luận và các yếu tố ảnh hưởng tính hợp lệ
  * 3.8. Kết luận chương
* DANH MỤC TÀI LIỆU THAM KHẢO

---

## DANH MỤC CÁC KÝ HIỆU, CÁC CHỮ VIẾT TẮT

| Chữ viết tắt | Thuật ngữ tiếng Anh | Nghĩa tiếng Việt |
| :--- | :--- | :--- |
| **ANN** | Approximate Nearest Neighbor | Tìm kiếm láng giềng gần nhất xấp xỉ |
| **BM25** | Best Matching 25 | Thuật toán xếp hạng xác suất BM25 |
| **DCG** | Discounted Cumulative Gain | Độ lợi tích lũy có chiết khấu |
| **DPR** | Dense Passage Retrieval | Truy xuất đoạn văn theo vector dày đặc |
| **IDF** | Inverse Document Frequency | Tần suất nghịch đảo tài liệu |
| **IR** | Information Retrieval | Truy xuất thông tin |
| **LLM** | Large Language Model | Mô hình ngôn ngữ lớn |
| **MRR** | Mean Reciprocal Rank | Trung bình nghịch đảo thứ hạng |
| **NDCG** | Normalized Discounted Cumulative Gain | Độ lợi tích lũy chiết khấu chuẩn hóa |
| **NFC** | Normalization Form C | Dạng chuẩn hóa Unicode tổ hợp |
| **qrels** | Query Relevance Judgments | Tập nhãn chân lý mức độ liên quan |
| **RAG** | Retrieval-Augmented Generation | Sinh văn bản tăng cường truy xuất |
| **RRF** | Reciprocal Rank Fusion | Dung hợp theo nghịch đảo thứ hạng |
| **TF-IDF** | Term Frequency – Inverse Document Frequency | Tần suất từ – tần suất nghịch đảo tài liệu |

---

## DANH MỤC CÁC BẢNG
* **Bảng 1.1.** Tổng hợp so sánh các công trình nghiên cứu liên quan
* **Bảng 2.1.** Cấu trúc trường dữ liệu của thực thể chunk tài liệu
* **Bảng 2.2.** Ma trận cấu hình thực nghiệm C1–C4
* **Bảng 3.1.** Môi trường và thiết lập thực nghiệm
* **Bảng 3.2.** Thống kê bộ benchmark theo nhóm, nguồn và tập
* **Bảng 3.3.** Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool
* **Bảng 3.4.** Tham số tối ưu chọn trên tập dev
* **Bảng 3.5.** Kết quả tổng thể trên tập test
* **Bảng 3.6.** Kết quả theo nhóm truy vấn trên tập test (RQ1)
* **Bảng 3.7.** So sánh ghép cặp có kiểm định thống kê (RQ2, RQ3, X2)
* **Bảng 3.8.** Độ trễ theo tầng xử lý (ms)
* **Bảng 3.9.** Kết quả theo nguồn câu hỏi (phân tích độ nhạy)
* **Bảng 3.10.** Phân bố truy vấn thất bại theo tầng và nguyên nhân
* **Bảng 3.11.** Kết quả các hướng khai thác X3–X6

---

## DANH MỤC CÁC HÌNH
* **Hình 1.1.** Kiến trúc tổng quan của một hệ thống RAG
* **Hình 1.2.** Quy trình truy xuất thông tin tổng quát
* **Hình 1.3.** Cơ chế hoạt động của Sparse Retrieval
* **Hình 1.4.** Cơ chế hoạt động của Dense Retrieval
* **Hình 1.5.** Phân loại các họ phương pháp truy xuất thông tin
* **Hình 1.6.** Kiến trúc truy xuất hai tầng kết hợp Cross-Encoder Reranker
* **Hình 3.1.** Ảnh hưởng của α đến chỉ số chính trên tập dev
* **Hình 3.2.** Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N
* **Hình 3.3.** Chỉ số chính theo nhóm truy vấn trên tập test

---

# CHƯƠNG 1: CƠ SỞ LÝ LUẬN VÀ TỔNG QUAN NGHIÊN CỨU

Chương này hệ thống hóa cơ sở lý luận của đề tài, bao gồm các khái niệm nền tảng về kiến trúc RAG, bài toán truy xuất thông tin và các họ phương pháp truy xuất hiện hành; đồng thời khảo sát tổng quan các công trình nghiên cứu trong nước và ngoài nước để xác định khoảng trống nghiên cứu mà đề tài hướng đến giải quyết. Toàn bộ nội dung trình bày trong chương là căn cứ khoa học trực tiếp cho các quyết định thiết kế hệ thống ở Chương 2.

## 1.1. Các khái niệm liên quan đến truy xuất tài liệu trong hệ thống RAG

### 1.1.1. Khái niệm mô hình RAG (Retrieval-Augmented Generation)

Retrieval-Augmented Generation (RAG) là kiến trúc kết hợp giữa hai thành phần: một mô-đun truy xuất thông tin (Retriever) và một mô hình ngôn ngữ lớn (Large Language Model – LLM) đóng vai trò sinh văn bản. Khác với cách tiếp cận sinh văn bản thuần túy (pure generation) vốn chỉ dựa vào tri thức đã được mã hóa trong tham số của mô hình tại thời điểm huấn luyện, RAG cho phép mô hình truy cập vào một nguồn tri thức ngoài (external knowledge base) được cập nhật độc lập, từ đó khắc phục hai hạn chế cố hữu của LLM: hiện tượng “ảo giác” (hallucination) và giới hạn về tính cập nhật của tri thức (knowledge cutoff) [1], [2], [3].

*Hình 1.1. Kiến trúc tổng quan của một hệ thống RAG*

Kiến trúc tổng quan của một hệ thống RAG gồm ba giai đoạn chính [1], [2]:
1. **Indexing (Lập chỉ mục):** tài liệu nguồn được phân đoạn thành các đơn vị nhỏ hơn (chunk), sau đó được mã hóa thành vector embedding và lưu trữ trong một cơ sở dữ liệu vector (Vector Database) như FAISS, Milvus, Qdrant, Chroma hay Pinecone [4], [5].
2. **Retrieval (Truy xuất):** khi có truy vấn (query) từ người dùng, hệ thống mã hóa truy vấn thành vector và thực hiện tìm kiếm các đoạn tài liệu có độ tương đồng cao nhất trong không gian vector, hoặc kết hợp với các phương pháp khớp từ vựng [2], [6].
3. **Generation (Sinh văn bản):** các đoạn tài liệu được truy xuất (context) cùng với truy vấn gốc được đưa vào LLM thông qua kỹ thuật prompt engineering, từ đó mô hình sinh câu trả lời có căn cứ (grounded) trên ngữ cảnh thực tế thay vì suy diễn từ tri thức nội tại [1], [2].

Nguyên lý quan trọng của RAG là bổ sung nguồn tri thức phi tham số bên ngoài cho tri thức đã được mã hóa trong tham số của mô hình. Nhờ đó, mô hình sinh có thể sử dụng đồng thời tri thức nội tại và ngữ cảnh được truy xuất để tạo câu trả lời. Điều này đặc biệt phù hợp với các bài toán có tính đặc thù cao về miền dữ liệu (domain-specific) như tài liệu học thuật, nơi tri thức thay đổi thường xuyên theo học kỳ, môn học hoặc chương trình đào tạo [1], [2].

### 1.1.2. Khái niệm truy xuất thông tin và các chỉ số đánh giá

Truy xuất thông tin (Information Retrieval – IR) là lĩnh vực nghiên cứu các phương pháp tìm kiếm những đơn vị dữ liệu (tài liệu, đoạn văn bản) liên quan nhất đến một truy vấn cho trước trong một tập hợp dữ liệu lớn [7]. Trong bối cảnh RAG, bài toán IR được thu hẹp thành bài toán khớp truy vấn – đoạn tài liệu (Query–Chunk Matching): với một câu hỏi $Q$ của người dùng và tập hợp các chunk $\{C_1, C_2, \dots, C_n\}$ đã được lập chỉ mục, hệ thống cần xếp hạng (rank) các chunk theo mức độ liên quan giảm dần, sao cho các chunk chứa thông tin trả lời cho $Q$ nằm ở vị trí cao nhất [2], [6].

*Hình 1.2. Quy trình truy xuất thông tin tổng quát*

Để đánh giá định lượng chất lượng của một hệ thống truy xuất, các độ đo phổ biến được sử dụng gồm:
* **Hit Rate@K:** tỷ lệ truy vấn mà trong top-K kết quả trả về có ít nhất một chunk đúng (relevant). Đây là độ đo trực quan, phản ánh khả năng “tìm thấy” thông tin đúng của hệ thống [7].
* **MRR (Mean Reciprocal Rank):** trung bình của nghịch đảo thứ hạng của chunk đúng đầu tiên xuất hiện trong danh sách kết quả, qua toàn bộ tập truy vấn. MRR nhạy với vị trí xếp hạng, do đó phản ánh tốt hơn Hit Rate về mức độ “gần đầu danh sách” của kết quả đúng [8].
* **NDCG@K (Normalized Discounted Cumulative Gain):** độ đo tính đến cả mức độ liên quan (relevance) theo thang điểm (không chỉ nhị phân đúng/sai) và vị trí xuất hiện trong danh sách, có chiết khấu logarit theo thứ hạng, phù hợp khi tài liệu có nhiều mức độ liên quan khác nhau [9].
* **Precision và Recall:** Precision đo tỷ lệ chunk liên quan trong số các chunk được truy xuất; Recall đo tỷ lệ chunk liên quan được truy xuất trên tổng số chunk liên quan thực sự tồn tại trong tập dữ liệu [7].

Việc lựa chọn độ đo phù hợp phụ thuộc vào đặc thù bài toán: với hệ thống hỏi–đáp tài liệu học tập, nơi câu trả lời thường phụ thuộc vào 1–3 đoạn tài liệu quan trọng nhất, Hit Rate@K và MRR thường được ưu tiên sử dụng làm chỉ số đánh giá chính.

### 1.1.3. Khái niệm truy xuất từ vựng (Sparse Retrieval)

Truy xuất từ vựng (Sparse Retrieval) là phương pháp truy xuất truyền thống, hoạt động dựa trên nguyên lý khớp từ khóa (keyword matching) giữa truy vấn và tài liệu, không thông qua biểu diễn ngữ nghĩa liên tục. Hai thuật toán tiêu biểu là:

*Hình 1.3. Cơ chế hoạt động của Sparse Retrieval*

* **TF-IDF (Term Frequency – Inverse Document Frequency):** đánh trọng số một từ trong tài liệu dựa trên tần suất xuất hiện của từ đó trong tài liệu (TF) và độ hiếm của từ đó trên toàn tập dữ liệu (IDF). Từ xuất hiện nhiều trong một tài liệu nhưng hiếm gặp trong toàn kho ngữ liệu sẽ có trọng số cao, phản ánh khả năng phân biệt tài liệu tốt [10], [11].
* **BM25 (Best Matching 25):** là biến thể cải tiến của TF-IDF, bổ sung cơ chế bão hòa tần suất từ (term frequency saturation) và chuẩn hóa theo độ dài tài liệu, giúp hạn chế việc các tài liệu dài chiếm ưu thế không hợp lý. BM25 hiện là thuật toán sparse retrieval được sử dụng phổ biến nhất trong các hệ thống tìm kiếm thực tế (ví dụ: Elasticsearch, Lucene) [12], [13].

Ưu điểm của Sparse Retrieval nằm ở khả năng tra cứu chính xác các thực thể có tính định danh cao: từ khóa chuyên ngành, mã môn học, mã hiệu văn bản, số liệu, tên riêng – những trường hợp mà việc khớp từ vựng chính xác (exact match) mang lại kết quả tin cậy hơn so với biểu diễn ngữ nghĩa. Ngoài ra, Sparse Retrieval không đòi hỏi huấn luyện mô hình, có chi phí tính toán thấp và khả năng diễn giải (interpretability) cao [12].

Nhược điểm chủ yếu là khả năng xử lý hạn chế đối với hiện tượng đồng nghĩa (synonymy) và biến thể diễn đạt (paraphrase): nếu truy vấn và tài liệu sử dụng từ ngữ khác nhau nhưng cùng ý nghĩa, Sparse Retrieval sẽ không thể khớp được, dẫn đến bỏ sót thông tin liên quan (low recall) trong các truy vấn diễn đạt tự nhiên [6], [12].

### 1.1.4. Khái niệm truy xuất ngữ nghĩa (Dense Retrieval)

Truy xuất ngữ nghĩa (Dense Retrieval) là phương pháp biểu diễn cả truy vấn và tài liệu dưới dạng các vector embedding có số chiều cố định trong một không gian ẩn (latent space) liên tục, sao cho các đoạn văn bản có ý nghĩa tương đồng sẽ có vector gần nhau về mặt khoảng cách (thường đo bằng cosine similarity hoặc dot product) [6].

*Hình 1.4. Cơ chế hoạt động của Dense Retrieval*

Kiến trúc phổ biến nhất cho Dense Retrieval là Bi-Encoder: truy vấn và tài liệu được mã hóa độc lập bởi hai encoder (có thể chia sẻ trọng số), cho phép tính toán trước (pre-compute) và lưu trữ embedding của toàn bộ tài liệu, từ đó việc tìm kiếm tại thời điểm truy vấn chỉ cần một phép tính encode cho truy vấn và tìm kiếm gần đúng (Approximate Nearest Neighbor – ANN) trong không gian vector [14], [15]. Các mô hình embedding tiêu biểu gồm:

* **DPR (Dense Passage Retrieval):** một trong những công trình đặt nền móng cho Dense Retrieval trong bài toán hỏi–đáp mở (Open-domain QA) [6].
* **OpenAI Embeddings (text-embedding-3 và các phiên bản kế tiếp):** các mô hình embedding thương mại, chất lượng cao, hỗ trợ đa ngôn ngữ [16].
* **BGE (BAAI General Embedding):** họ mô hình embedding mã nguồn mở có hiệu năng cạnh tranh, hỗ trợ tốt các ngôn ngữ trong đó có tiếng Việt thông qua các biến thể đa ngôn ngữ như BGE-M3 hoặc các bản fine-tune riêng [17], [18].

Ưu điểm nổi bật của Dense Retrieval là khả năng hiểu ngữ cảnh và nắm bắt ý nghĩa của câu hỏi tự nhiên, xử lý tốt các trường hợp đồng nghĩa, diễn đạt lại (paraphrase) hoặc truy vấn không trùng từ khóa chính xác với tài liệu [6].

Nhược điểm là hiệu năng suy giảm đối với các truy vấn đòi hỏi khớp chính xác thực thể định danh (mã số, thuật ngữ hiếm, ký hiệu chuyên ngành) do bản chất “làm mờ” thông tin khi nén văn bản thành vector có số chiều cố định; đồng thời đòi hỏi chi phí tính toán và lưu trữ lớn hơn, cũng như phụ thuộc vào chất lượng dữ liệu huấn luyện của mô hình embedding [6], [19].

### 1.1.5. Khái niệm truy xuất hỗn hợp (Hybrid Retrieval) và tái xếp hạng (Reranking)

Hybrid Retrieval là phương pháp kết hợp đồng thời Sparse Retrieval và Dense Retrieval nhằm tận dụng ưu điểm của cả hai: khả năng khớp từ khóa chính xác của BM25 và khả năng hiểu ngữ nghĩa của Dense Retrieval [20]. Hai kỹ thuật dung hợp điểm số phổ biến gồm:

*Hình 1.5. Phân loại các họ phương pháp truy xuất thông tin*

* **Weighted Sum (tổng có trọng số):** điểm số cuối cùng của một chunk được tính bằng tổng có trọng số giữa điểm sparse và điểm dense đã được chuẩn hóa (ví dụ: min–max normalization), theo công thức dạng:
  $$\text{score} = \alpha \times \text{score\_sparse} + (1 - \alpha) \times \text{score\_dense}$$
  trong đó $\alpha$ là siêu tham số cần được tối ưu hóa thực nghiệm [20].
* **Reciprocal Rank Fusion (RRF):** thay vì dung hợp trực tiếp điểm số (vốn có thang đo khác nhau giữa hai phương pháp), RRF dung hợp dựa trên thứ hạng (rank) của mỗi chunk trong từng danh sách kết quả riêng lẻ, theo công thức:
  $$\text{RRF\_score}(d) = \sum \frac{1}{k + \text{rank}_i(d)}$$
  với $k$ là hằng số làm mượt (thường $k = 60$). RRF có ưu điểm không cần chuẩn hóa thang điểm và ổn định hơn trong thực nghiệm [21].

Sau bước Hybrid Retrieval, một tập ứng viên (candidate set) gồm top-N chunk được đưa qua bước Reranking (tái xếp hạng) sử dụng mô hình Cross-Encoder. Khác với Bi-Encoder (mã hóa độc lập truy vấn và tài liệu), Cross-Encoder nhận đồng thời cặp (truy vấn, chunk) làm đầu vào và tính điểm liên quan trực tiếp thông qua cơ chế attention chéo giữa hai chuỗi, cho phép mô hình nắm bắt tương tác ngữ nghĩa chi tiết hơn. Do chi phí tính toán cao (không thể pre-compute), Cross-Encoder chỉ được áp dụng trên tập ứng viên đã được thu hẹp từ bước truy xuất ban đầu, đóng vai trò “tinh chỉnh” thứ tự xếp hạng cuối cùng trước khi đưa vào LLM [22].

*Hình 1.6. Kiến trúc truy xuất hai tầng kết hợp Cross-Encoder Reranker*

### 1.1.6. Khái niệm và đặc thù của dữ liệu tài liệu học tập

Dữ liệu tài liệu học tập (giáo trình, slide bài giảng, đề thi, tài liệu tham khảo) có những đặc thù riêng biệt so với văn bản tổng quát, ảnh hưởng trực tiếp đến việc thiết kế hệ thống truy xuất:
* **Cấu trúc phân cấp phức tạp:** giáo trình thường được tổ chức theo chương – mục – tiểu mục, trong khi slide bài giảng có cấu trúc rời rạc theo từng trang, dẫn đến thách thức trong việc phân đoạn (chunking) sao cho vẫn giữ được ngữ cảnh liên kết [23].
* **Thuật ngữ chuyên ngành và mã hiệu:** tài liệu học thuật chứa nhiều mã môn học, ký hiệu công thức, thuật ngữ kỹ thuật có tính định danh cao – đây là nhóm thông tin mà Dense Retrieval dễ bỏ sót nhưng Sparse Retrieval xử lý tốt [19].
* **Tính đa dạng định dạng:** tài liệu học tập kết hợp văn bản xuôi, công thức toán học, bảng biểu, hình ảnh minh họa, đòi hỏi các phương pháp tiền xử lý và trích xuất riêng biệt trước khi đưa vào pipeline lập chỉ mục [24].
* **Tính đặc thù ngôn ngữ:** đối với tài liệu tiếng Việt, các vấn đề như phân tách từ (word segmentation), dấu thanh và sự thiếu hụt tương đối của các mô hình embedding được huấn luyện chuyên biệt cho tiếng Việt học thuật càng làm gia tăng thách thức truy xuất [25], [26].

Những đặc thù trên là cơ sở lý luận cho việc đề xuất áp dụng Hybrid Retrieval kết hợp Reranking – thay vì chỉ sử dụng Dense Retrieval đơn lẻ – nhằm đảm bảo hệ thống vừa hiểu được câu hỏi tự nhiên của người học, vừa không bỏ sót các mã hiệu, thuật ngữ chính xác cần thiết.

## 1.2. Tổng quan nghiên cứu trong nước và ngoài nước

### 1.2.1. Các nghiên cứu trên thế giới

**Nền tảng RAG và sự tiến hóa của phương pháp truy xuất.** Kể từ công trình đặt nền móng của Lewis và cộng sự (2020) dựa trên Dense Retrieval đơn lẻ [1], các nghiên cứu tiếp theo đã chỉ ra tính phụ thuộc mạnh mẽ của hiệu năng truy xuất vào đặc thù dữ liệu. Trong khi Dense Retrieval (Karpukhin và cộng sự, 2020; Lyu và cộng sự, 2024) vượt trội ở các câu hỏi suy luận và tổng hợp đa tài liệu, thì BM25 lại chiếm ưu thế ở miền dữ liệu chứa nhiều mã định danh, số liệu và thuật ngữ chính xác (Strich và cộng sự, 2026). Điều này khẳng định không có một phương pháp đơn lẻ nào tối ưu cho mọi miền tri thức [19], [27].

**Hiệu quả của Hybrid Retrieval và Reranking.** Việc dung hợp Sparse và Dense Retrieval qua Reciprocal Rank Fusion giúp cải thiện vượt bậc độ phủ (Recall) so với các phương pháp thành phần (Sawarkar và cộng sự, 2024; Strich và cộng sự, 2026). Đặc biệt, việc bổ sung Cross-Encoder Reranker sau bước Hybrid đóng vai trò quyết định trong việc tối ưu độ chính xác (Precision) ở top đầu, giúp tăng mạnh các chỉ số MRR và Recall (Lyu và cộng sự, 2024; Strich và cộng sự, 2026) [19], [20], [27].

**Ứng dụng trong giáo dục.** Điển hình là hệ thống MARK (Lian, 2026), việc tích hợp Hybrid Retrieval và Reranker đã giải quyết đồng thời cả nhu cầu tra cứu từ khóa hành chính lẫn câu hỏi khái niệm chuyên ngành, đồng thời chứng minh kỹ thuật giữ lại tiêu đề mục (section heading) trong chunk giúp tối ưu hóa chất lượng truy xuất tài liệu bài giảng [28].

### 1.2.2. Các nghiên cứu tại Việt Nam

**Xu hướng ứng dụng RAG tiếng Việt.** Giai đoạn 2025–2026 ghi nhận sự bùng nổ của RAG cho tiếng Việt, song chủ yếu tập trung vào tư vấn tuyển sinh (Nguyen và cộng sự, HCMUT, 2025), dịch vụ công (La và cộng sự, với Vistral-7B) hoặc pháp lý (Agentic RAG). Các hệ thống này đa phần sử dụng Dense Retrieval đơn lẻ kết hợp kiến trúc phân tầng (Rule-based/FAQ) hoặc tác tử để giảm tải chi phí, chưa đi sâu vào tối ưu hóa tầng truy xuất văn bản học thuật chuyên sâu [29], [30].

**Thách thức với dữ liệu học thuật.** Nghiên cứu chatbot Toán học của Phạm Văn Khánh và Phạm Vũ Anh Tuấn (sử dụng BGE-M3, Milvus) đã chỉ rõ các rào cản lớn: hiện tượng ảo giác, thiếu tri thức chuẩn hóa theo chương trình đào tạo và khó khăn khi xử lý dữ liệu đa định dạng (công thức, ký hiệu) [24].

**Phương pháp luận và hướng tiếp cận mới.** Nghiên cứu của Nguyen và Nguyen (2026) về sinh tự động bộ QA đã đặt nền móng chuẩn hóa benchmark cho RAG tiếng Việt. Trong khi đó, các hướng tiếp cận dùng đồ thị tri thức (Knowledge Graph – Bui và cộng sự) vẫn ở giai đoạn sơ khởi, khẳng định hướng đi tối ưu hóa Hybrid Retrieval kết hợp Reranking là giải pháp thực tế và khả thi nhất cho tài liệu học tập hiện nay.

**Bảng 1.1. Tổng hợp so sánh các công trình nghiên cứu liên quan**

| Công trình | Phương pháp truy xuất | Phát hiện chính | Khoảng trống so với đề tài |
| :--- | :--- | :--- | :--- |
| Lewis và cộng sự (2020) [1] | Dense Retrieval đơn lẻ | Đặt nền móng kiến trúc RAG | Không dùng Hybrid, không thử nghiệm trên dữ liệu học thuật |
| Strich và cộng sự (2026) [19] | BM25, Dense, Hybrid RRF, Rerank | Hybrid + Rerank vượt trội mọi phương pháp đơn lẻ; Rerank là thành phần cải thiện lớn nhất | Miền tài chính tiếng Anh, không phải tài liệu học tập tiếng Việt |
| Lyu và cộng sự (2024) – CRUD-RAG [27] | BM25, Dense, Hybrid + Rerank | Dense vượt BM25 ở câu hỏi suy luận đa tài liệu; Hybrid + Rerank cải thiện toàn diện | Tiếng Trung, không có đặc thù mã môn học/thuật ngữ học thuật |
| Lian (2026) – MARK [28] | Hybrid (BM25 + FAISS Dense) + Rerank | Hybrid cần thiết để xử lý cả câu hỏi hành chính và khái niệm | Tiếng Anh, chưa tối ưu hóa trọng số dung hợp một cách hệ thống |
| Nguyen và cộng sự (2025) – URAG [29] | FAQ rule-based + RAG (Dense) | Kiến trúc hai tầng giảm chi phí, tăng độ chính xác | Chưa thực nghiệm Hybrid/Rerank ở tầng truy xuất tài liệu |
| Phung – Chatbot dịch vụ công (VN) [30] | Dense Retrieval (Vistral-7B) | Cải thiện độ chính xác so với LLM thuần túy | Chưa so sánh Sparse/Hybrid, không phải miền học thuật |
| Phạm V.K. & Phạm V.A.T. – Chatbot Toán (VN) [24] | RAG + BGE-M3 embedding | Nêu rõ thách thức công thức/hình ảnh trong tài liệu học thuật | Chưa thực nghiệm Hybrid Retrieval, tập trung vào một môn học |

### 1.2.3. Nhận xét và khoảng trống nghiên cứu (Research Gap)

Từ việc phân tích so sánh các công trình trong nước và quốc tế nêu trên, có thể rút ra một số nhận xét sau.

Điểm mạnh của các nghiên cứu đã công bố: (1) đã lượng hóa cụ thể bằng số liệu thực nghiệm rằng Hybrid Retrieval kết hợp Reranking vượt trội các phương pháp đơn lẻ trên nhiều miền dữ liệu khác nhau (tài chính, thương mại điện tử, dữ liệu tổng quát), với mức cải thiện có thể lên tới hàng chục điểm phần trăm ở các chỉ số như MRR@3 và Recall@5 (Strich và cộng sự, 2026); (2) đã chỉ ra rằng hiệu quả tương đối giữa Sparse và Dense Retrieval phụ thuộc vào đặc thù dữ liệu – dữ liệu càng chứa nhiều mã định danh, ký hiệu, thuật ngữ chính xác thì Sparse Retrieval càng có vai trò quan trọng, điều này có ý nghĩa trực tiếp cho tài liệu học tập vốn chứa nhiều mã môn học và ký hiệu công thức; (3) tại Việt Nam, các công trình như URAG, chatbot Toán học và chatbot pháp lý đã bước đầu chứng minh tính khả thi của RAG cho dữ liệu tiếng Việt chuyên ngành [24], [29].

Hạn chế chung: các công trình quốc tế lượng hóa rõ hiệu quả của Hybrid Retrieval kết hợp Reranking đều thực nghiệm trên dữ liệu tiếng Anh hoặc tiếng Trung, miền tài chính/thương mại điện tử – chưa có công trình nào thực nghiệm trực tiếp trên tài liệu học thuật tiếng Việt; các công trình tại Việt Nam (URAG, chatbot dịch vụ công, chatbot Toán học) đều dùng Dense Retrieval là chủ đạo, không thực nghiệm so sánh định lượng với Sparse Retrieval hoặc Hybrid Retrieval, và cũng không áp dụng bước Reranking – trong khi chính các nghiên cứu quốc tế (Strich và cộng sự, 2026; Lyu và cộng sự, 2024) đã chỉ ra Reranking là thành phần mang lại cải thiện lớn nhất. Duy nhất công trình MARK (Lian, 2026) triển khai đầy đủ Hybrid + Rerank cho giáo dục, nhưng thực hiện trên tiếng Anh và không tối ưu hóa hệ thống trọng số dung hợp [24], [28], [30].

Từ đó, có thể xác định khoảng trống nghiên cứu (research gap) như sau: (1) chưa có công trình nào thực nghiệm định lượng Hybrid Retrieval kết hợp Reranking trên tập tài liệu học tập tiếng Việt (giáo trình, slide, đề thi) – nơi hội tụ đồng thời hai đặc thù mà các nghiên cứu quốc tế đã chỉ ra là quan trọng cho việc lựa chọn phương pháp truy xuất: mật độ cao các mã định danh/thuật ngữ chuyên ngành (đòi hỏi Sparse Retrieval) và tính đa dạng của câu hỏi tự nhiên từ người học (đòi hỏi Dense Retrieval); (2) các nghiên cứu RAG giáo dục tại Việt Nam hiện nay chưa tối ưu hóa một cách hệ thống trọng số dung hợp giữa Sparse và Dense Retrieval ($\alpha$ trong Weighted Sum hoặc hằng số $k$ trong RRF), cũng như chưa đánh giá vai trò của Reranker trên dữ liệu tiếng Việt học thuật; (3) thiếu vắng một bộ chỉ số đánh giá định lượng (Hit Rate@K, MRR, NDCG@K) được áp dụng nhất quán để so sánh trực tiếp các cấu hình truy xuất khác nhau trên cùng một tập tài liệu học tập tiếng Việt. Đây chính là khoảng trống mà đề tài nghiên cứu này hướng đến giải quyết [24], [29].

## 1.3. Định hướng nghiên cứu của đề tài

### 1.3.1. Mục tiêu nghiên cứu
Xây dựng và tối ưu hóa quy trình Hybrid Retrieval kết hợp Reranking nhằm nâng cao hiệu quả truy xuất cho tập tài liệu học tập, làm nền tảng cho hệ thống hỏi–đáp dựa trên RAG phục vụ người học.

### 1.3.2. Nhiệm vụ nghiên cứu
1. Nghiên cứu cơ sở lý thuyết về Sparse Retrieval, Dense Retrieval, Hybrid Retrieval và Reranking trong kiến trúc RAG [20], [21].
2. Xây dựng bộ dữ liệu thực nghiệm gồm tài liệu học tập (giáo trình, slide bài giảng, đề thi) và tập truy vấn tương ứng.
3. Thực nghiệm so sánh hiệu năng giữa các mô hình đơn lẻ (BM25, Dense Retrieval) và mô hình đề xuất (Hybrid Retrieval kết hợp Reranker) [20], [22].
4. Đo lường và phân tích mức độ cải thiện thông qua các chỉ số Hit Rate@K, MRR, NDCG@K giữa các phương pháp [7], [9].
5. Đề xuất cấu hình tối ưu (trọng số dung hợp, phương pháp Reranker) phù hợp với đặc thù tài liệu học tập tiếng Việt.

### 1.3.3. Đối tượng và phạm vi nghiên cứu
* **Đối tượng nghiên cứu:** các kỹ thuật truy xuất thông tin (Sparse, Dense, Hybrid) và tái xếp hạng (Reranking) trong kiến trúc RAG.
* **Phạm vi dữ liệu:** tài liệu học tập bao gồm giáo trình, slide bài giảng, đề thi thuộc một hoặc nhiều môn học cụ thể, cùng tập truy vấn câu hỏi mô phỏng nhu cầu tra cứu thực tế của người học.
* **Phạm vi kỹ thuật:** giới hạn trong phạm vi giai đoạn Retrieval của pipeline RAG (không đi sâu vào tối ưu hóa mô hình sinh văn bản/Generation), tập trung vào việc đánh giá và tối ưu hóa chất lượng ngữ cảnh được truy xuất trước khi đưa vào LLM.

## 1.4. Kết luận chương

Chương 1 đã hệ thống hóa cơ sở lý luận của bài toán truy xuất tài liệu học tập trong kiến trúc RAG: từ khái niệm RAG và ba giai đoạn Indexing – Retrieval – Generation, đến bộ độ đo đánh giá truy xuất chuẩn mực và ba họ phương pháp truy xuất: Sparse, Dense và Hybrid kết hợp Reranking. Phân tích cho thấy mỗi họ phương pháp đơn lẻ đều có vùng thất bại riêng, trong khi đặc thù của học liệu tiếng Việt – cấu trúc phân cấp, mật độ mã hiệu cao, đa định dạng và đặc thù phân tách từ – đòi hỏi đồng thời cả năng lực khớp từ vựng lẫn năng lực hiểu ngữ nghĩa.

Khảo sát tổng quan các công trình trong nước và quốc tế đã xác định được khoảng trống nghiên cứu rõ ràng: thiếu vắng thực nghiệm định lượng Hybrid Retrieval kết hợp Reranking trên học liệu tiếng Việt, thiếu việc tối ưu hóa có hệ thống trọng số dung hợp và thiếu một bộ chỉ số đánh giá được áp dụng nhất quán. Đây là căn cứ trực tiếp cho mục tiêu, nhiệm vụ và phạm vi nghiên cứu đã xác lập ở mục 1.3, đồng thời là tiền đề để Chương 2 tiến hành phân tích yêu cầu, thiết kế kiến trúc và xây dựng hệ thống truy xuất cụ thể.

---

# CHƯƠNG 2: PHÂN TÍCH, THIẾT KẾ VÀ XÂY DỰNG HỆ THỐNG TRUY XUẤT TÀI LIỆU HỌC TẬP BẰNG HYBRID RETRIEVAL

Trên cơ sở lý luận và khoảng trống nghiên cứu đã xác định ở Chương 1, chương này tiến hành phân tích yêu cầu bài toán, mô hình hóa bài toán truy xuất, thiết kế kiến trúc tổng thể hệ thống, xây dựng bộ dữ liệu thực nghiệm, triển khai các cấu hình truy xuất đối chứng và thiết lập phương pháp đánh giá định lượng cho toàn bộ quá trình thực nghiệm.

## 2.1. Phân tích yêu cầu bài toán nghiên cứu

### 2.1.1. Phân tích đối tượng sử dụng và nhu cầu truy xuất
Hệ thống được thiết kế hướng tới việc giải quyết bài toán tìm kiếm và khai thác tri thức chính xác từ kho học liệu số tiếng Việt trong mô hình RAG. Các nhóm tác nhân (actors) chính tham gia tương tác với hệ thống bao gồm:
* **Người học (người dùng trực tiếp):** bao gồm sinh viên, học viên có nhu cầu tra cứu bài học, ôn thi, giải đáp khái niệm. Người học đưa ra phổ truy vấn cực kỳ đa dạng: từ các từ khóa định danh cụ thể (mã môn học, thuật ngữ hiếm, công thức) đòi hỏi exact-match, cho đến các câu hỏi mở, giải thích nguyên lý hoặc câu hỏi diễn đạt tự do đòi hỏi khả năng thấu hiểu ngữ nghĩa sâu.
* **Người quản trị / giảng viên (người dùng gián tiếp):** chịu trách nhiệm quản lý kho tài liệu số, cập nhật phiên bản giáo trình, kiểm duyệt nội dung. Nhóm này đòi hỏi thông tin metadata phân cấp chi tiết (tên tài liệu, chương, mục, trang) để đảm bảo tính minh bạch và khả năng truy vết nguồn gốc câu trả lời độc lập với LLM.
* **Nhu cầu nghiên cứu thực nghiệm:** đòi hỏi hệ thống có khả năng chạy lặp lại các tập truy vấn thử nghiệm trên nhiều cấu hình thuật toán khác nhau, lưu trữ chi tiết danh sách xếp hạng kèm điểm số phân rã và độ trễ để tính toán các chỉ số định lượng (Hit Rate@K, MRR, NDCG@K).

### 2.1.2. Phân tích bối cảnh và giới hạn của hệ thống
**Bối cảnh RAG.** Pipeline RAG chuẩn gồm ba giai đoạn: lập chỉ mục (Indexing), truy xuất (Retrieval) và sinh văn bản (Generation). Trong đó, tầng Retrieval đóng vai trò quyết định trần chất lượng (upper bound) của toàn bộ hệ thống. Nếu tầng truy xuất trả về các chunk sai hoặc xếp các căn cứ đúng ở vị trí quá thấp (ngoài cửa sổ ngữ cảnh), LLM sẽ thiếu căn cứ thực tế và không thể tránh khỏi hiện tượng ảo giác (hallucination). Các nghiên cứu về cách LLM sử dụng ngữ cảnh dài cũng cho thấy mô hình khai thác tốt nhất thông tin nằm ở đầu ngữ cảnh và suy giảm rõ rệt với thông tin nằm ở giữa, do đó việc đưa căn cứ đúng lên top đầu – chứ không chỉ đưa được nó vào context – là yêu cầu bắt buộc [31].

**Đặc thù kho tài liệu học tập.** Kho học liệu là sự pha trộn của nhiều hình thức biểu đạt có cấu trúc khác nhau:
* **Giáo trình:** chứa văn bản dài, cấu trúc chương mục chặt chẽ, giàu tính lý thuyết và định nghĩa hình thức.
* **Slide bài giảng:** nội dung cô đọng, trình bày dạng bullet points rời rạc, phụ thuộc mật thiết vào tiêu đề slide để tạo ngữ cảnh trọn vẹn.
* **Đề thi và đáp án:** chứa câu hỏi ngắn, bài tập tính toán, ký hiệu công thức và đáp án định danh.

**Giới hạn phạm vi kỹ thuật.** Đề tài tập trung chuyên sâu vào tầng Retrieval và Reranking. Mô hình sinh ngôn ngữ (LLM) được giữ cố định ở bên ngoài phạm vi đánh giá nhằm triệt tiêu các biến số thiên lệch từ mô hình sinh, đảm bảo kết quả đo lường phản ánh chính xác năng lực của các thuật toán truy xuất.

### 2.1.3. Câu hỏi nghiên cứu và giả thuyết kiểm định
**RQ1:** BM25 và Dense Retrieval thể hiện hiệu năng khác nhau như thế nào trên các nhóm truy vấn tài liệu học tập tiếng Việt?
* *Giả thuyết:* BM25 vượt trội ở các truy vấn định danh, ký hiệu, thuật ngữ chính xác; Dense Retrieval vượt trội ở các câu hỏi khái niệm, suy luận và câu hỏi được diễn đạt lại (paraphrase) [19].
* *Bằng chứng:* đo lường Hit Rate@K, MRR, NDCG@K trên toàn bộ tập truy vấn và phân rã theo từng nhóm truy vấn chức năng.

**RQ2:** Dung hợp kết quả Sparse và Dense Retrieval (Hybrid Retrieval) có cải thiện có ý nghĩa thống kê so với cấu hình đơn lẻ tốt nhất không?
* *Giả thuyết:* Hybrid Retrieval kết hợp được ưu thế từ vựng và ngữ nghĩa, đạt kết quả trung bình cao hơn có ý nghĩa thống kê so với cấu hình đơn lẻ tốt nhất [20].
* *Bằng chứng:* so sánh ghép cặp trên cùng tập truy vấn và kiểm định khoảng tin cậy bootstrap [32].

**RQ3:** Reranking cải thiện thứ hạng top đầu thêm bao nhiêu và đổi lại chi phí độ trễ (latency) như thế nào?
* *Giả thuyết:* Cross-Encoder Reranker cải thiện vượt trội các chỉ số MRR và NDCG ở Top-K đầu tiên nhờ cơ chế full self-attention, nhưng làm tăng độ trễ tính toán [22].
* *Bằng chứng:* đối chiếu tỷ lệ tăng trưởng MRR/NDCG trước – sau khi Rerank với thời gian xử lý truy vấn (ms).

### 2.1.4. Yêu cầu chức năng và phi chức năng
Hệ thống được thiết kế đáp ứng chặt chẽ các yêu cầu chức năng sau:
* **FR-01 (Quản lý dữ liệu nguồn):** tiếp nhận tài liệu đa định dạng, gán định danh duy nhất (doc_id), trích xuất văn bản và lưu trữ metadata phân cấp chi tiết.
* **FR-02 (Tiền xử lý và chunking):** làm sạch văn bản, chuẩn hóa tiếng Việt Unicode NFC, phân đoạn văn bản thành các chunk (chunk_id) có bảo tồn ngữ cảnh tiêu đề phân cấp.
* **FR-03 (Lập chỉ mục song song):** xây dựng đồng bộ chỉ mục nghịch đảo BM25 (Sparse) và chỉ mục vector FAISS/Milvus (Dense) từ cùng một tập chunk chuẩn tắc.
* **FR-04 (Vận hành truy xuất đa cấu hình):** hỗ trợ thực thi độc lập BM25, Dense Retrieval, Hybrid (RRF/Weighted Sum) và Hybrid kết hợp Cross-Encoder Reranker.
* **FR-05 (Quản lý benchmark):** quản lý tập truy vấn thử nghiệm và nhãn chân lý qrels đa mức với mã phiên bản cố định.
* **FR-06 (Tính toán và báo cáo):** tự động tính toán các chỉ số định lượng và kết xuất nhật ký thực nghiệm chi tiết từng lượt truy vấn (run logs).

Song song đó, hệ thống phải đáp ứng các yêu cầu phi chức năng:
* **Độ chính xác:** chunk liên quan phải được xếp ở các vị trí đầu danh sách (đo bằng Hit Rate@K, MRR, NDCG@K).
* **Hiệu năng và độ trễ:** Dense Retrieval phản hồi nhanh trên chỉ mục vector; Cross-Encoder chỉ xử lý trên tập ứng viên Top-N giới hạn để kiểm soát độ trễ trong ngưỡng cho phép.
* **Khả năng mở rộng:** thiết kế module hóa cho phép dễ dàng cắm/rút mô hình embedding, hàm dung hợp hoặc mô hình reranker mới.
* **Tính ổn định và tái lập (reproducibility):** cố định random seed, phiên bản mô hình, chuẩn hóa Unicode NFC thống nhất; phân tách tệp tham số cấu hình độc lập khỏi mã nguồn.
* **Khả năng truy nguyên (traceability):** mọi kết quả trả về phải gắn liền với chunk_id, tên tài liệu, vị trí chương mục và số trang tương ứng.

## 2.2. Phân tích bài toán truy xuất tài liệu học tập

### 2.2.1. Mô hình hóa bài toán
Gọi kho tài liệu gồm $n$ đoạn văn bản (chunk) là:
$$R = \{C_1, C_2, \dots, C_n\}$$

Tập hợp $m$ truy vấn của người dùng là:
$$Q = \{q_1, q_2, \dots, q_m\}$$

Với mỗi truy vấn $q \in Q$, hệ thống truy xuất sử dụng hàm tính điểm liên quan $S(q, C_i)$ hoặc cơ chế dung hợp thứ hạng để sắp xếp các chunk theo thứ tự giảm dần và trích xuất danh sách Top-K kết quả:
$$R_k(q) = \{C_{(1)}, C_{(2)}, \dots, C_{(k)}\}$$
trong đó $C_{(1)}$ có điểm số/thứ hạng cao nhất.

Mức độ liên quan trong tập nhãn chân lý $R_q$ được lượng hóa theo thang đo đa mức (graded relevance) – cách tiếp cận chuẩn mực trong đánh giá truy xuất hiện đại [9]:
* **Mức 0 (không liên quan):** hoàn toàn không chứa thông tin phục vụ trả lời câu hỏi.
* **Mức 1 (liên quan bổ trợ):** chứa thông tin bối cảnh chung, khái niệm bổ trợ nhưng chưa đủ trả lời trọn vẹn.
* **Mức 2 (căn cứ trực tiếp):** chứa thông tin chính xác, đầy đủ căn cứ trực tiếp để trả lời trọn vẹn câu hỏi.

### 2.2.2. Đặc thù dữ liệu và phân nhóm truy vấn tiếng Việt học thuật
Nhằm kiểm định toàn diện năng lực của các thuật toán, tập truy vấn benchmark được phân chia thành bốn nhóm chức năng rõ rệt:
* **Nhóm định danh / chính xác (exact-match):** chứa mã môn học, tên thuật ngữ kỹ thuật hiếm, ký hiệu toán học, số điều mục. Đây là nhóm bài test đo lường năng lực khớp từ vựng chính xác của BM25 và kiểm tra rủi ro “làm mờ thông tin” của Dense Retrieval.
* **Nhóm khái niệm (conceptual):** hỏi về định nghĩa, nguyên lý vận hành, ý nghĩa của khái niệm học thuật. Người học thường diễn đạt bằng câu chữ khác biệt với định nghĩa hình thức trong sách giáo trình; đây là thước đo năng lực thấu hiểu ngữ nghĩa của Dense Retrieval.
* **Nhóm diễn đạt lại (paraphrase):** câu hỏi cố tình sử dụng từ đồng nghĩa, hoán đổi cấu trúc câu hoặc diễn đạt gián tiếp. Điểm số từ vựng của BM25 sẽ giảm mạnh; đây là bài test đánh giá sự phối hợp giữa Dense Retrieval và Cross-Encoder Reranker.
* **Nhóm tổng hợp / đa đoạn (multi-chunk):** câu hỏi yêu cầu so sánh, đối chiếu hoặc tổng hợp kiến thức trải dài trên nhiều phần, nhiều slide khác nhau. Đòi hỏi độ phủ (Recall) rất cao để không bỏ sót các đoạn căn cứ; phục vụ đánh giá năng lực mở rộng của Hybrid Retrieval [27].

## 2.3. Thiết kế tổng thể hệ thống

### 2.3.1. Nguyên tắc thiết kế
* **Tách biệt các tầng xử lý (separation of concerns):** các module tiền xử lý, lập chỉ mục, Sparse Retrieval, Dense Retrieval, Fusion, Reranking và Context Builder được tách biệt hoàn toàn.
* **Tính công bằng trong thực nghiệm (fairness):** mọi cấu hình C1–C4 đều được kiểm thử trên cùng một tập chunk, cùng tập truy vấn và cùng bộ nhãn chân lý qrels.
* **Tính module hóa (plug-and-play):** cho phép cắm/rút linh hoạt các mô hình embedding mới hoặc thuật toán dung hợp khác mà không phải sửa đổi cấu trúc hệ thống.
* **Chuẩn hóa dữ liệu:** toàn bộ dữ liệu trung gian trao đổi giữa các tầng được chuẩn hóa theo định dạng JSON Lines thống nhất.

### 2.3.2. Kiến trúc tổng thể và luồng xử lý dữ liệu
Hệ thống gồm các thành phần cốt lõi sau:
1. **Kho tài liệu học tập:** lưu trữ giáo trình, slide bài giảng, đề thi và tài liệu tham khảo.
2. **Module tiền xử lý:** trích xuất text, làm sạch Unicode, bảo toàn metadata và phân đoạn thành các chunk ngữ nghĩa.
3. **Sparse Retrieval:** sử dụng BM25 tạo danh sách kết quả dựa trên tần suất từ vựng và IDF.
4. **Dense Retrieval:** sử dụng mô hình Bi-Encoder biểu diễn truy vấn và chunk dưới dạng vector trong không gian ẩn, tìm kiếm qua Cosine Similarity.
5. **Hybrid Retrieval:** dung hợp hai danh sách kết quả bằng thuật toán RRF hoặc Weighted Sum.
6. **Cross-Encoder Reranker:** nhận truy vấn và tập ứng viên Hybrid để tính điểm tương tác trực tiếp qua self-attention và tái xếp hạng.
7. **Context Builder:** trích chọn Top-K chunk đứng đầu sau reranking, đóng gói thành context tối ưu.
8. **LLM (Generation):** nhận câu hỏi và context đã tinh lọc để sinh câu trả lời (nằm ngoài phạm vi đánh giá).

Hệ thống vận hành theo hai tuyến:
* **Tuyến ngoại tuyến (Offline Indexing Pipeline):** gồm tiền xử lý, phân đoạn và xây dựng song song Sparse Index cùng Dense Index.
* **Tuyến trực tuyến (Online Retrieval Pipeline):** thực hiện khi nhận truy vấn – đồng thời tra cứu BM25 và Dense, dung hợp kết quả (Hybrid), tái xếp hạng qua Cross-Encoder và xuất Top-K context.

### 2.3.3. Thiết kế luồng xử lý dữ liệu chi tiết
**a) Luồng xây dựng chỉ mục (Offline Indexing)**
Tài liệu → Trích xuất nội dung → Làm sạch và chuẩn hóa NFC → Phân đoạn Structure-aware Chunking → Gán doc_id, chunk_id và metadata phân cấp.
Từ cùng tập chunk $C$, hệ thống đồng thời tạo:
$C \rightarrow \text{Sparse Inverted Index (cho BM25)}$
$C \rightarrow \text{Dense Vector Index (cho FAISS/Milvus qua Bi-Encoder)}$

**b) Luồng xử lý truy vấn (Online Retrieval)**
Với truy vấn $Q$ từ người dùng, hệ thống phát song song hai nhánh:
$$R_s(Q) = \text{BM25\_Retrieve}(Q, \text{Sparse\_Index})$$
$$R_d(Q) = \text{Dense\_Retrieve}(Q, \text{Dense\_Index})$$
Hai danh sách được dung hợp tại tầng Hybrid:
$$R_h(Q) = \text{Fusion}(R_s(Q), R_d(Q)) \quad \text{[RRF hoặc Weighted Sum]}$$
Tập ứng viên Top-N được đưa qua tầng tái xếp hạng:
$$R_r(Q) = \text{Cross\_Encoder\_Rerank}(Q, R_h(Q))$$
Cuối cùng, hệ thống trích xuất Top-K chunk làm context đưa sang LLM.

### 2.3.4. Thiết kế giao tiếp giữa các thành phần
Thực thể dữ liệu trung tâm xuyên suốt hệ thống là chunk tài liệu, được cấu trúc hóa với các trường thông tin chuẩn mực như trình bày ở Bảng 2.1.

**Bảng 2.1. Cấu trúc trường dữ liệu của thực thể chunk tài liệu**

| Trường | Ý nghĩa |
| :--- | :--- |
| `chunk_id` | Định danh của chunk |
| `document_id` | Định danh tài liệu nguồn |
| `content` | Nội dung chunk |
| `metadata` | Thông tin chương, mục, trang hoặc ngữ cảnh |
| `sparse_score` | Điểm BM25 |
| `dense_score` | Điểm Dense |
| `fusion_score` | Điểm Hybrid |
| `rerank_score` | Điểm Reranker |
| `rank` | Thứ hạng kết quả |

## 2.4. Xây dựng và xử lý bộ dữ liệu thực nghiệm

### 2.4.1. Thu thập và chuẩn hóa tài liệu
Kho dữ liệu bao gồm giáo trình chuyên ngành Công nghệ thông tin, slide bài giảng môn học và các bộ đề thi trắc nghiệm/tự luận tiếng Việt. Toàn bộ văn bản thô được đưa qua module làm sạch: chuyển đổi thống nhất sang chuẩn Unicode NFC (tránh lỗi font tổ hợp thường gặp trong tài liệu tiếng Việt cũ), loại bỏ các ký tự điều khiển ẩn, chuẩn hóa khoảng trắng và dấu câu [33].

### 2.4.2. Phân đoạn tài liệu (Structure-aware Chunking)
Thay vì cắt văn bản theo kích thước cửa sổ cố định (Fixed-size Chunking) vốn làm đứt gãy ngữ cảnh giữa các đoạn, đề tài áp dụng kỹ thuật Structure-aware Chunking – hướng tiếp cận đã được chứng minh là cải thiện đáng kể chất lượng truy xuất so với phân đoạn cơ học [23]:
* **Phân đoạn theo cấu trúc:** tận dụng cấu trúc chương – mục – tiểu mục của giáo trình và từng trang slide để xác định ranh giới cắt tự nhiên.
* **Bảo tồn tiêu đề ngữ cảnh:** mỗi chunk được tự động chèn thêm chuỗi metadata phân cấp vào phần đầu theo mẫu: `[Tên tài liệu] > [Chương] > [Mục] > Nội dung văn bản`. Kỹ thuật này giúp các mô hình embedding và BM25 không bị mất ngữ cảnh gốc ngay cả khi đoạn văn bản bên dưới là các gạch đầu dòng ngắn [28].

### 2.4.3. Xây dựng tập benchmark và kiểm soát rủi ro sai lệch
* **Chống thiên vị pooling:** tập nhãn qrels được hình thành bằng phương pháp pooling đa nguồn – hợp nhất kết quả từ BM25, nhiều mô hình Dense Retrieval và tìm kiếm thủ công để các chuyên gia dán nhãn, ngăn ngừa triệt để rủi ro thiên vị một phương pháp cụ thể (pooling bias) [34].
* **Phân tách tập dữ liệu:** bộ dữ liệu câu hỏi được phân tách nghiêm ngặt thành Development Set (dùng để hiệu chỉnh và tối ưu hóa siêu tham số $\alpha, k$) và Test Set (hoàn toàn cô lập, chỉ dùng để đánh giá báo cáo cuối cùng) nhằm loại bỏ rủi ro rò rỉ dữ liệu (data leakage).

## 2.5. Xây dựng các phương pháp truy xuất và Hybrid Retrieval

### 2.5.1. Cấu hình đối chứng Sparse Retrieval (BM25)
Hệ thống triển khai thuật toán Okapi BM25 chuẩn mực [12], [13]. Điểm số được tính toán dựa trên tần suất xuất hiện của từ khóa trong chunk, tần suất nghịch đảo tài liệu (IDF) và tỷ lệ độ dài chunk so với độ dài trung bình toàn kho tài liệu. Siêu tham số $k_1 \in [1.2, 1.5]$ (kiểm soát tốc độ bão hòa tần suất từ) và $b = 0.75$ (kiểm soát mức độ phạt tài liệu dài) được đóng băng cố định. Trước khi tra cứu chỉ mục nghịch đảo, văn bản truy vấn và tài liệu được tách từ tiếng Việt bằng công cụ chuyên dụng [26].

### 2.5.2. Cấu hình đối chứng Dense Retrieval
Cấu hình này sử dụng mô hình Bi-Encoder tiên tiến hỗ trợ tối ưu tiếng Việt. Truy vấn và chunk được mã hóa thành các vector nhúng độc lập trong không gian liên tục 768/1024 chiều. Độ tương đồng ngữ nghĩa được xác định qua hàm Cosine Similarity. Việc tìm kiếm láng giềng gần nhất được tăng tốc thông qua thư viện FAISS/Milvus [4], [5], [18].

### 2.5.3. Cấu hình Hybrid Retrieval và phương pháp dung hợp
Module Hybrid tiếp nhận đồng thời hai danh sách Top-N ứng viên từ BM25 và Dense Retrieval, sau đó dung hợp theo hai cơ chế:
* **Reciprocal Rank Fusion (RRF):** $\text{RRF}(d) = \sum \frac{1}{c + \text{rank}_i(d)}$ với $c = 60$. Ưu điểm là không cần giả định đồng nhất thang điểm [21].
* **Weighted Sum:** $\text{Score}(d) = \alpha \times \hat{s}_{\text{sparse}}(d) + (1 - \alpha) \times \hat{s}_{\text{dense}}(d)$ sau khi chuẩn hóa Min–Max về đoạn $[0, 1]$ [20].

### 2.5.4. Cấu hình tái xếp hạng (Cross-Encoder Reranking)
Tập ứng viên Top-N (N = 25–50) sau bước Hybrid được đưa vào mô hình Cross-Encoder chuyên dụng. Mô hình này nhận chuỗi ghép cặp `[CLS] + Query + [SEP] + Chunk + [EOS]` và thực thi cơ chế full self-attention xuyên suốt tất cả các tầng biểu diễn. Nhờ vậy, mô hình nhận biết chính xác mối liên hệ tương tác từ vựng và ngữ cảnh sâu sắc, gán lại điểm số liên quan thực sự để sắp xếp lại danh sách Top-K cuối cùng [22].

### 2.5.5. Ma trận cấu hình thực nghiệm và kiểm định loại trừ (Ablation Study)
Để đánh giá tường tận vai trò đóng góp của từng cấu phần, hệ thống thiết lập ma trận bốn cấu hình thực nghiệm chuẩn hóa như trình bày ở Bảng 2.2.

**Bảng 2.2. Ma trận cấu hình thực nghiệm C1–C4**

| Mã cấu hình | Truy xuất ban đầu | Cơ chế dung hợp | Tái xếp hạng | Vai trò đối chứng |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | BM25 | Không | Không | Đường cơ sở Sparse |
| **C2** | Dense Retrieval | Không | Không | Đường cơ sở Dense |
| **C3** | BM25 + Dense | RRF / Weighted Sum | Không | Đo lường hiệu quả dung hợp |
| **C4** | BM25 + Dense | RRF / Weighted Sum | Cross-Encoder | Đóng góp của Reranker |

*Ghi chú:* C3 và C4 được triển khai thành hai biến thể theo cơ chế dung hợp: C3-RRF/C3-WS và C4-RRF/C4-WS. Ngoài ra, cấu hình đối chứng X1 (Dense + Reranker) được bổ sung để tách riêng đóng góp của Reranker khỏi đóng góp của bước dung hợp (xem mục 3.6).

## 2.6. Thiết kế thực nghiệm và phương pháp đánh giá

### 2.6.1. Quy trình thực nghiệm chuẩn tắc
Quy trình đánh giá gồm năm bước tuần tự nghiêm ngặt:
* **Bước 1 (đóng băng tài nguyên):** đóng băng phiên bản tài liệu học tập, cố định random seed và siêu tham số mô hình.
* **Bước 2 (khởi tạo chỉ mục đồng bộ):** tạo đồng thời BM25 Inverted Index và Dense Vector Index trên cùng tập chunk.
* **Bước 3 (tối ưu tham số trên Dev Set):** thử nghiệm các giá trị trọng số $\alpha$ (Weighted Sum) và hằng số $c$ (RRF) trên Development Set để chọn ra bộ siêu tham số tốt nhất.
* **Bước 4 (đánh giá chính thức trên Test Set):** chạy bốn cấu hình C1–C4 trên Test Set độc lập, thu thập bảng xếp hạng kết quả.
* **Bước 5 (tổng hợp thống kê và error analysis):** tính toán các chỉ số định lượng, kiểm định ý nghĩa thống kê bằng phương pháp bootstrap và phân tích lỗi trên các truy vấn thất bại [32].

### 2.6.2. Các chỉ số đánh giá định lượng
Hệ thống đo lường bộ chỉ số định lượng tiêu chuẩn quốc tế [7], [9]:
* **Hit Rate@K:** tỷ lệ câu hỏi mà trong Top-K kết quả xuất hiện ít nhất một đoạn tài liệu đúng.
  $$\text{HitRate@K} = \frac{1}{|Q|} \sum I(L_{q,K} \cap R_q \neq \emptyset)$$
* **MRR (Mean Reciprocal Rank):** nghịch đảo thứ hạng của đoạn tài liệu đúng đầu tiên xuất hiện trong danh sách trả về [8].
  $$\text{MRR} = \frac{1}{|Q|} \sum \frac{1}{\text{rank}_q}$$
* **NDCG@K:** đo lường chất lượng xếp hạng có tính đến mức độ liên quan đa mức (0, 1, 2) và chiết khấu logarit theo vị trí [9].
  $$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
* **Recall@K:** tỷ lệ đoạn tài liệu đúng được tìm thấy trong Top-K trên tổng số đoạn đúng thực tế tồn tại trong qrels.
* **Độ trễ truy vấn (latency):** đo lường thời gian xử lý trung bình và phân vị P95 (tính bằng mili-giây – ms) cho từng lượt truy vấn.

## 2.7. Triển khai và tích hợp hệ thống RAG

### 2.7.1. Kiến trúc triển khai tổng thể
Hệ thống được triển khai theo kiến trúc module hóa tách rời (decoupled architecture). Kho tài liệu học liệu được xử lý và lập chỉ mục trước. Khi có truy vấn từ người học, Retrieval Service tiếp nhận và thực thi chuỗi xử lý: Query → Sparse + Dense → Hybrid → Reranking → Context. Context tối ưu sau đó mới được chuyển sang cho LLM sinh câu trả lời. Kiến trúc này cho phép tối ưu hóa tầng truy xuất một cách độc lập mà không cần tái cấu trúc mô hình sinh.

### 2.7.2. Triển khai module lập chỉ mục (Indexing Module)
Module lập chỉ mục chịu trách nhiệm toàn bộ quá trình Offline Pipeline: Tài liệu → Trích xuất văn bản → Tiền xử lý → Chunking → Metadata Extraction. Sau đó, dữ liệu chunk được nạp đồng thời vào Sparse Index (BM25) và Dense Index (Vector DB). Quá trình này giúp triệt tiêu chi phí tính toán embedding văn bản tại thời điểm người dùng truy vấn.

### 2.7.3. Triển khai dịch vụ Retrieval (Retrieval Service)
Dịch vụ Retrieval đảm nhiệm việc tiếp nhận truy vấn trực tuyến, thực thi song song BM25 và Dense Search, hợp nhất kết quả qua hàm dung hợp, kích hoạt Cross-Encoder Reranker và trả về Top-K chunk kèm đầy đủ metadata, thứ hạng và điểm số phục vụ đánh giá.

### 2.7.4. Tích hợp với pipeline RAG
Pipeline hoàn chỉnh vận hành theo chu trình khép kín: Query → Retrieval → Hybrid → Reranking → Context → LLM → Answer. Trọng tâm cốt lõi của nghiên cứu là chặng từ Retrieval đến Reranking, bảo đảm context đưa vào prompt luôn có độ chính xác và độ liên quan cao nhất.

### 2.7.5. Tối ưu hiệu năng
* **Tối ưu Dense Retrieval:** vector embedding của toàn bộ chunk được tính trước và lưu trên chỉ mục vector tối ưu. Khi có truy vấn, hệ thống chỉ cần tính đúng một vector cho câu hỏi và thực hiện truy vấn ANN [15].
* **Tối ưu Reranking:** Cross-Encoder chỉ được gọi trên tập ứng viên Top-N (N = 25–50) trích từ bước Hybrid thay vì quét toàn bộ kho tài liệu. Kiến trúc hai tầng (two-stage) này giúp cân bằng giữa chất lượng xếp hạng và thời gian đáp ứng thời gian thực [22].

### 2.7.6. Khả năng mở rộng và bảo trì
Nhờ thiết kế module hóa, hệ thống cho phép thay thế độc lập từng thành phần: thay mô hình embedding mới chỉ cần rebuild lại Dense Index; thử nghiệm thuật toán Reranker hoặc phương pháp dung hợp mới mà không ảnh hưởng tới dữ liệu nguồn hay mô hình LLM phía sau.

## 2.8. Kết luận chương
Trong Chương 2, đề tài đã hoàn thành phân tích toàn diện yêu cầu và đặc thù bài toán truy xuất học liệu số tiếng Việt. Trên cơ sở đó, kiến trúc hệ thống Hybrid Retrieval kết hợp Cross-Encoder Reranker hai tầng đã được thiết kế và mô hình hóa chi tiết.

Về dữ liệu, đề tài đã xây dựng quy trình tiền xử lý, chuẩn hóa Unicode NFC và phân đoạn thông minh Structure-aware Chunking có bảo tồn ngữ cảnh tiêu đề. Về phương pháp luận, ma trận thực nghiệm bốn cấu hình C1–C4 cùng hệ thống chỉ số đánh giá chuẩn mực (Hit Rate@K, MRR, NDCG@K) đã được định hình rõ ràng, kèm theo các biện pháp kiểm soát rủi ro sai lệch thực nghiệm như chống thiên vị pooling và phân tách Dev/Test Set.

Toàn bộ thiết kế hệ thống và cơ sở thực nghiệm này tạo tiền đề khoa học vững chắc để triển khai các kịch bản kiểm thử, so sánh số liệu thực nghiệm định lượng và phân tích kết quả chuyên sâu trong chương tiếp theo.

---

# CHƯƠNG 3: THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

Chương này trình bày quá trình triển khai thực nghiệm theo quy trình chuẩn tắc đã thiết kế ở mục 2.6: mô tả môi trường và bộ dữ liệu, tinh chỉnh siêu tham số trên tập dev, đánh giá chính thức các cấu hình C1–C4 trên tập test để trả lời ba câu hỏi nghiên cứu RQ1–RQ3, phân tích lỗi, và trình bày các hướng khai thác mở rộng X1–X6. Toàn bộ số liệu trong chương được sinh tự động bởi công cụ `python -m src.cli eval …`; mỗi bảng/hình ghi rõ tệp nguồn để bảo đảm khả năng tái lập.

## 3.1. Môi trường và thiết lập thực nghiệm

Thực nghiệm được thực hiện trên [[ĐIỀN: nền tảng và GPU, ví dụ Google Colab Pro – GPU … — nguồn: runs/<RUN_ID>/latency.json, mục hardware]]. Các thành phần được cố định như sau: mô hình embedding BAAI/bge-m3 (1024 chiều, chuẩn hóa L2, tìm kiếm chính xác bằng tích vô hướng), mô hình reranker BAAI/bge-reranker-v2-m3, BM25 với k1 = 1,5 và b = 0,75, mỗi nhánh truy xuất trả về top-L = 100 ứng viên, số chunk đưa vào ngữ cảnh K = 5, seed = 42. Kho tài liệu được phân đoạn theo cấu trúc (structure-aware) với tối đa 350 từ mỗi chunk, chồng lấn 50 từ, và gắn tiền tố `[Tài liệu] > [Chương] > [Mục]`.

**Bảng 3.1. Môi trường và thiết lập thực nghiệm**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_1.md (lệnh `python -m src.cli eval report`)]]

## 3.2. Bộ dữ liệu và benchmark

### 3.2.1. Kho tài liệu học tập

Kho tài liệu gồm [[ĐIỀN: số tài liệu, loại (giáo trình/slide/đề thi), môn học, tổng số trang — nguồn: trang Documents / data/processed/<version>/manifest.json]], sau phân đoạn thu được [[ĐIỀN: số chunk — nguồn: data/indexes/<version>/index_meta.json]] chunk.

### 3.2.2. Xây dựng tập truy vấn

Tập truy vấn được xây dựng theo hai nguồn. (1) Câu hỏi nháp do mô hình ngôn ngữ [[ĐIỀN: tên mô hình LLM — nguồn: data/benchmark/drafts.jsonl, trường generator]] sinh từ các chunk được lấy mẫu phân tầng theo môn học, loại tài liệu và tài liệu, với câu lệnh riêng cho bốn nhóm truy vấn (mục 2.2.2). Mỗi câu hỏi nháp phải vượt qua kiểm tra tự động: nhóm định danh phải chứa mã/ký hiệu có trong chunk; nhóm diễn đạt lại phải có độ trùng từ vựng (Jaccard trên từ nội dung) với chunk nguồn không quá 0,2; đoạn trích căn cứ phải xuất hiện nguyên văn trong chunk. Sau đó thành viên nhóm rà soát từng câu (giữ/sửa/loại). (2) Câu hỏi do người viết trực tiếp mà không nhìn tài liệu, nhằm mô phỏng câu hỏi thực của người học và giảm thiên lệch từ vựng của câu hỏi sinh tự động. Các câu gần trùng lặp (cosine ≥ 0,92) được loại bỏ. Tập cuối cùng gồm [[ĐIỀN: tổng số câu hỏi, số câu mỗi nhóm, tỉ lệ câu do người viết — nguồn: data/benchmark/benchmark_description.json]].

**Bảng 3.2. Thống kê bộ benchmark theo nhóm, nguồn và tập**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_2.md]]

### 3.2.3. Gán nhãn mức độ liên quan và độ tin cậy

Để chống thiên vị pooling (mục 2.4.3), với mỗi câu hỏi, tập ứng viên được hợp nhất từ top-15 của sáu hệ thống (C1, C2, C3-RRF, C3-WS, C4-WS, X1), chunk nguồn và các chunk do người dán nhãn tự tìm thêm. Tệp dán nhãn được xáo trộn và ẩn tên hệ thống. Hai người dán nhãn độc lập theo thang 0/1/2 trên [[ĐIỀN: số cặp/tỉ lệ câu hỏi được dán nhãn đôi — nguồn: data/benchmark/agreement.json]]; độ đồng thuận Cohen's κ có trọng số bậc hai đạt [[ĐIỀN: κ — nguồn: data/benchmark/agreement.json]], [[ĐIỀN: diễn giải theo thang Landis–Koch]]. Các trường hợp bất đồng được cả nhóm thống nhất lần cuối. Mỗi nhãn liên quan kèm đoạn trích căn cứ, cho phép ánh xạ nhãn sang các cách phân đoạn khác (mục 3.6.4).

**Bảng 3.3. Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_3.md]]

[[ĐIỀN: nhận xét về số chunk liên quan chỉ một hệ thống tìm thấy — bằng chứng pooling đa nguồn là cần thiết]]

### 3.2.4. Phân tách dev/test

Tập câu hỏi được chia phân tầng theo nhóm truy vấn thành dev (30%) và test (70%) với seed 42; câu hỏi không có chunk liên quan bị loại ([[ĐIỀN: số câu bị loại — nguồn: kết quả lệnh `bench split`]]). Mã băm của tệp truy vấn, qrels và tham số tối ưu được ghi vào `benchmark_manifest.json`; công cụ đánh giá ghi nhận mọi thay đổi tham số sau lần chạy test đầu tiên (trạng thái khóa: [[ĐIỀN: có/không vi phạm — nguồn: runs/<RUN_ID>/config.json, trường lock]]).

## 3.3. Tinh chỉnh siêu tham số trên tập dev

Chỉ số chính MRR@10 được khai báo trước khi thực nghiệm. Trên tập dev, hệ thống quét α ∈ {0; 0,1; …; 1} cho Weighted Sum, k ∈ {10, 20, 40, 60, 100} cho RRF, β ∈ {0,1; 0,2; 0,3; 0,5} cho dung hợp thích nghi và N ∈ {10, 20, 30, 50} cho reranker; khi hòa điểm, giá trị gần mặc định được chọn.

**Bảng 3.4. Tham số tối ưu chọn trên tập dev**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_4.md hoặc data/benchmark/frozen_params.yaml]]

**Hình 3.1. Ảnh hưởng của α đến chỉ số chính trên tập dev**

[[ĐIỀN: chèn hình — nguồn: runs/<RUN_ID>/report/figure_3_1.png]]

[[ĐIỀN: nhận xét — α tối ưu có khác nhau giữa các nhóm truy vấn không (ví dụ nhóm định danh ưa α cao, nhóm diễn đạt lại ưa α thấp)? Đây là căn cứ cho hướng X2]]

## 3.4. Kết quả trên tập test

**Bảng 3.5. Kết quả tổng thể trên tập test**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_5.md]]

### 3.4.1. RQ1 – So sánh BM25 và Dense Retrieval theo nhóm truy vấn

**Bảng 3.6. Kết quả theo nhóm truy vấn trên tập test (RQ1)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_6.md]]

**Hình 3.3. Chỉ số chính theo nhóm truy vấn trên tập test**

[[ĐIỀN: chèn hình — nguồn: runs/<RUN_ID>/report/figure_3_3.png]]

[[ĐIỀN: kết luận giả thuyết RQ1 — BM25 có vượt trội ở nhóm định danh và Dense ở nhóm khái niệm/diễn đạt lại không; nêu chênh lệch và khoảng tin cậy theo nhóm từ comparisons.csv]]

### 3.4.2. RQ2 – Hiệu quả của Hybrid Retrieval

**Bảng 3.7. So sánh ghép cặp có kiểm định thống kê (RQ2, RQ3, X2)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_7.md (lệnh `python -m src.cli eval compare` trước `eval report`)]]

[[ĐIỀN: kết luận giả thuyết RQ2 — C3-RRF/C3-WS so với cấu hình đơn lẻ tốt nhất (chọn trên dev): chênh lệch MRR@10, CI 95% bootstrap, p đã hiệu chỉnh Holm]]

### 3.4.3. RQ3 – Đóng góp và chi phí của Reranker

**Bảng 3.8. Độ trễ theo tầng xử lý (ms)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_8.md]]

**Hình 3.2. Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N**

[[ĐIỀN: chèn hình — nguồn: runs/<RUN_ID>/report/figure_3_2.png]]

[[ĐIỀN: kết luận giả thuyết RQ3 — mức tăng MRR@10/NDCG@10 của C4 so với C3 và của X1 so với C2 (Bảng 3.7), đổi lại độ trễ tăng bao nhiêu ms (trung bình và P95)]]

### 3.4.4. Phân tích độ nhạy theo nguồn câu hỏi

**Bảng 3.9. Kết quả theo nguồn câu hỏi (phân tích độ nhạy)**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_9.md]]

[[ĐIỀN: thứ hạng các cấu hình có giữ nguyên giữa câu hỏi do LLM sinh và câu hỏi do người viết không? Nếu có, kết luận của RQ1–RQ3 vững hơn trước thiên lệch của câu hỏi sinh tự động]]

## 3.5. Phân tích lỗi

Với cấu hình mục tiêu C4-WS, các truy vấn không có chunk liên quan nào trong top-10 được phân loại tự động theo tầng gây lỗi dựa trên điểm phân rã (Bảng 2.1): (i) *first_stage_miss* – cả BM25 và Dense đều không đưa chunk đúng vào top-L; (ii) *fusion_demoted* – chunk đúng có trong top-L nhưng bị đẩy ra ngoài top-N sau dung hợp; (iii) *rerank_demoted* – chunk đúng có trong top-N nhưng bị Reranker hạ xuống ngoài top-10. Một mẫu tối đa 50 truy vấn thất bại được nhóm gắn nhãn nguyên nhân thủ công: lỗi trích xuất/OCR, chunk cắt ngang ý, tách từ, lệch từ vựng, cần suy luận nhiều bước, nhãn qrels sai, khác.

**Bảng 3.10. Phân bố truy vấn thất bại theo tầng và nguyên nhân**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_10.md (lệnh `eval errors`, gắn nhãn cột cause trong error_sample.csv, rồi `eval errors --summarize`)]]

[[ĐIỀN: 2–3 ví dụ lỗi tiêu biểu kèm câu hỏi, chunk đúng, chunk xếp đầu và nguyên nhân]]

## 3.6. Các hướng khai thác mở rộng

Phần này trình bày các thí nghiệm bổ sung nhằm khai thác các khoảng trống nghiên cứu đã nêu ở mục 1.2.3, vượt ra ngoài ma trận C1–C4.

### 3.6.1. X1 – Dense + Reranker: tách đóng góp của Reranker

So sánh C4 với C2 gộp chung hai tác động (dung hợp và tái xếp hạng). Cấu hình X1 áp dụng Reranker trực tiếp lên Dense Retrieval, cho phép tách riêng: đóng góp của Reranker (X1 so với C2) và giá trị gia tăng của dung hợp khi đã có Reranker (C4 so với X1). [[ĐIỀN: kết quả — nguồn: Bảng 3.5 và Bảng 3.7]]

### 3.6.2. X2 – Dung hợp thích nghi theo truy vấn

Thay vì một α cố định, α được điều chỉnh theo đặc trưng từ vựng của truy vấn: α = clamp(α₀ + β·(2·s − 1), 0,1, 0,9), trong đó s ∈ [0, 1] tổng hợp tỉ lệ chữ số, tỉ lệ ký hiệu, sự hiện diện của mã định danh và IDF lớn nhất (chuẩn hóa) của các từ trong truy vấn. Truy vấn giàu định danh nhận α lớn (ưu tiên BM25), truy vấn ngữ nghĩa nhận α nhỏ (ưu tiên Dense). Hướng này trực tiếp giải quyết khoảng trống (2) về tối ưu hóa trọng số dung hợp. [[ĐIỀN: kết quả X2/X2-R so với C3-WS/C4-WS — nguồn: Bảng 3.7]]

### 3.6.3. X3 – Ảnh hưởng của tách từ tiếng Việt đến BM25

So sánh tách từ theo khoảng trắng (âm tiết) với tách từ ghép bằng pyvi và VnCoreNLP [26] cho C1 và C3-WS. [[ĐIỀN: kết quả — nguồn: Bảng 3.11]]

### 3.6.4. X4 – Structure-aware chunking và tiền tố tiêu đề

So sánh phân đoạn theo cấu trúc với phân đoạn cửa sổ cố định (450 từ, chồng lấn 75), và bật/tắt tiền tố `[Tài liệu] > [Chương] > [Mục]`. Nhãn qrels được ánh xạ tự động sang từng cách phân đoạn thông qua đoạn trích căn cứ, nên không cần dán nhãn lại. [[ĐIỀN: kết quả — nguồn: Bảng 3.11]]

### 3.6.5. X5 – Số ứng viên đưa vào Reranker

Quét N ∈ {10, 20, 30, 50} để xác định điểm cân bằng giữa chất lượng và độ trễ. [[ĐIỀN: kết quả — nguồn: Hình 3.2 và Bảng 3.11]]

### 3.6.6. X6 – Độ nhạy theo mô hình embedding

Thay BGE-M3 bằng [[ĐIỀN: tên mô hình embedding tiếng Việt được chọn]] cho C2 và C4-WS để kiểm tra kết luận có phụ thuộc vào mô hình embedding hay không. [[ĐIỀN: kết quả — nguồn: Bảng 3.11]]

**Bảng 3.11. Kết quả các hướng khai thác X3–X6**

[[ĐIỀN: dán bảng — nguồn: runs/<RUN_ID>/report/table_3_11.md]]

## 3.7. Thảo luận và các yếu tố ảnh hưởng tính hợp lệ

### 3.7.1. Thảo luận

[[ĐIỀN: tổng hợp câu trả lời RQ1–RQ3, đối chiếu với các công trình ở Bảng 1.1 (Strich và cộng sự, 2026; Lyu và cộng sự, 2024; Lian, 2026), và đề xuất cấu hình tối ưu cho học liệu tiếng Việt (nhiệm vụ 5, mục 1.3.2)]]

### 3.7.2. Các yếu tố ảnh hưởng tính hợp lệ

* **Nguồn câu hỏi:** phần lớn câu hỏi được sinh bởi LLM từ chính các chunk nên có thể thiên về khớp từ vựng; ảnh hưởng này được kiểm tra bằng phân tích độ nhạy theo nguồn câu hỏi (Bảng 3.9) và kiểm tra độ trùng từ vựng với nhóm diễn đạt lại.
* **Cỡ mẫu theo nhóm:** mỗi nhóm truy vấn trên tập test có khoảng [[ĐIỀN: số câu mỗi nhóm]] câu, nên khoảng tin cậy theo nhóm rộng; kết luận theo nhóm chỉ mang tính định hướng.
* **Độ sâu pooling:** chunk liên quan nằm ngoài top-15 của mọi hệ thống và không được tìm thủ công sẽ không có nhãn, có thể làm giảm Recall tuyệt đối (nhưng tác động như nhau lên các cấu hình).
* **Phạm vi dữ liệu:** thực nghiệm trên [[ĐIỀN: số môn học]] môn học của một cơ sở đào tạo; khả năng khái quát sang môn học/cơ sở khác cần được kiểm chứng.
* **Phần cứng:** độ trễ đo trên GPU của Colab có biến động giữa các phiên; số liệu độ trễ dùng để so sánh tương đối giữa các cấu hình trong cùng một lần chạy.

## 3.8. Kết luận chương

[[ĐIỀN: tóm tắt kết quả chính của RQ1–RQ3, cấu hình khuyến nghị, đóng góp của các hướng khai thác X1–X6 và hạn chế còn lại]]

---

# DANH MỤC TÀI LIỆU THAM KHẢO

Mỗi tài liệu dưới đây được gắn liên kết trực tiếp tới trang công bố chính thức (DOI, ACL Anthology, IEEE Xplore hoặc arXiv). Các số trích dẫn [n] trong thân bài cũng là liên kết, bấm vào sẽ nhảy đến đúng mục tương ứng ở đây.

1. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledge-intensive NLP tasks,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 9459–9474. [Online]. Available: https://arxiv.org/abs/2005.11401
2. Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, “Retrieval-augmented generation for large language models: A survey,” arXiv preprint arXiv:2312.10997, 2024. [Online]. Available: https://arxiv.org/abs/2312.10997
3. Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, “Survey of hallucination in natural language generation,” ACM Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023. [Online]. Available: https://doi.org/10.1145/3571730
4. J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021. [Online]. Available: https://doi.org/10.1109/TBDATA.2019.2921572
5. J. Wang, X. Yi, R. Guo, H. Jin, P. Xu, S. Li, X. Wang, X. Guo, C. Li, X. Xu, K. Yu, Y. Yuan, Y. Zou, J. Long, Y. Cai, Z. Li, Z. Zhang, Y. Mo, J. Gu, R. Jiang, Y. Wei, and C. Xie, “Milvus: A purpose-built vector data management system,” in Proc. 2021 ACM SIGMOD International Conference on Management of Data, 2021, pp. 2614–2627. [Online]. Available: https://doi.org/10.1145/3448016.3457550
6. V. Karpukhin, B. Oğuz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih, “Dense passage retrieval for open-domain question answering,” in Proc. 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 6769–6781. [Online]. Available: https://aclanthology.org/2020.emnlp-main.550/
7. C. D. Manning, P. Raghavan, and H. Schütze, Introduction to Information Retrieval. Cambridge, U.K.: Cambridge University Press, 2008. [Online]. Available: https://nlp.stanford.edu/IR-book/
8. E. M. Voorhees, “The TREC-8 question answering track report,” in Proc. 8th Text REtrieval Conference (TREC-8), Gaithersburg, MD, USA, 1999, pp. 77–82. [Online]. Available: https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf
9. K. Järvelin and J. Kekäläinen, “Cumulated gain-based evaluation of IR techniques,” ACM Transactions on Information Systems, vol. 20, no. 4, pp. 422–446, 2002. [Online]. Available: https://doi.org/10.1145/582415.582418
10. K. Spärck Jones, “A statistical interpretation of term specificity and its application in retrieval,” Journal of Documentation, vol. 28, no. 1, pp. 11–21, 1972. [Online]. Available: https://doi.org/10.1108/eb026526
11. G. Salton and C. Buckley, “Term-weighting approaches in automatic text retrieval,” Information Processing & Management, vol. 24, no. 5, pp. 513–523, 1988. [Online]. Available: https://doi.org/10.1016/0306-4573(88)90021-0
12. S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,” Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009. [Online]. Available: https://doi.org/10.1561/1500000019
13. S. E. Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, and M. Gatford, “Okapi at TREC-3,” in Proc. 3rd Text REtrieval Conference (TREC-3), Gaithersburg, MD, USA, 1994, pp. 109–126. [Online]. Available: https://trec.nist.gov/pubs/trec3/t3_proceedings.html
14. N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in Proc. 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982–3992. [Online]. Available: https://aclanthology.org/D19-1410/
15. Y. A. Malkov and D. A. Yashunin, “Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 42, no. 4, pp. 824–836, 2020. [Online]. Available: https://doi.org/10.1109/TPAMI.2018.2889473
16. A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, J. Heidecke, P. Shyam, B. Power, T. E. Nekoul, G. Sastry, G. Krueger, D. Schnurr, F. P. Such, K. Hsu, M. Thompson, T. Khan, T. Sherbakov, J. Jang, P. Welinder, and L. Weng, “Text and code embeddings by contrastive pre-training,” arXiv preprint arXiv:2201.10005, 2022. [Online]. Available: https://arxiv.org/abs/2201.10005
17. S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y. Nie, “C-Pack: Packed resources for general Chinese embeddings,” in Proc. 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2024, pp. 641–649. [Online]. Available: https://doi.org/10.1145/3626772.3657878
18. J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 2318–2335. [Online]. Available: https://aclanthology.org/2024.findings-acl.137/
19. J. Strich, E. K. Isgorur, M. Trescher, C. Biemann, and M. Semmann, “T²-RAGBench: Text-and-table benchmark for evaluating retrieval-augmented generation,” in Proc. 19th Conference of the European Chapter of the Association for Computational Linguistics, 2026, pp. 165–191. [Online]. Available: https://aclanthology.org/2026.eacl-long.8/
20. K. Sawarkar, A. Mangal, and S. R. Solanki, “Blended RAG: Improving RAG (retriever-augmented generation) accuracy with semantic search and hybrid query-based retrievers,” arXiv preprint arXiv:2404.07220, 2024. [Online]. Available: https://arxiv.org/abs/2404.07220
21. G. V. Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank fusion outperforms Condorcet and individual rank learning methods,” in Proc. 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2009, pp. 758–759. [Online]. Available: https://doi.org/10.1145/1571941.1572114
22. R. Nogueira and K. Cho, “Passage re-ranking with BERT,” arXiv preprint arXiv:1901.04085, 2019. [Online]. Available: https://arxiv.org/abs/1901.04085
23. A. Jimeno Yepes, Y. You, J. Milczek, S. Laverde, and R. Li, “Financial report chunking for effective retrieval augmented generation,” arXiv preprint arXiv:2402.05131, 2024. [Online]. Available: https://arxiv.org/abs/2402.05131
24. P. V. Khanh and P. V. A. Tuan, “Building a Vietnamese math chatbot based on RAG and LLM: System design, implementation and experimental evaluation,” HNUE Journal of Science: Journal of Natural Sciences, 2025. [Online]. Available: https://hnuejs.edu.vn/ns/article/view/1284
25. D. Q. Nguyen and A. T. Nguyen, “PhoBERT: Pre-trained language models for Vietnamese,” in Findings of the Association for Computational Linguistics: EMNLP 2020, 2020, pp. 1037–1042. [Online]. Available: https://aclanthology.org/2020.findings-emnlp.92/
26. T. Vu, D. Q. Nguyen, D. Q. Nguyen, M. Dras, and M. Johnson, “VnCoreNLP: A Vietnamese natural language processing toolkit,” in Proc. 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations, 2018, pp. 56–60. [Online]. Available: https://aclanthology.org/N18-5012/
27. Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen, “CRUD-RAG: A comprehensive Chinese benchmark for retrieval-augmented generation of large language models,” ACM Transactions on Information Systems, 2024. [Online]. Available: https://doi.org/10.1145/3701228
28. Y. Lian, “Machine assistant with reliable knowledge: Enhancing student learning via RAG-based retrieval,” arXiv preprint arXiv:2506.23026, 2025. [Online]. Available: https://arxiv.org/abs/2506.23026
29. L. Nguyen and T. Quan, “URAG: Implementing a unified hybrid RAG for precise answers in university admission chatbots – A case study at HCMUT,” arXiv preprint arXiv:2501.16276, 2025. [Online]. Available: https://arxiv.org/abs/2501.16276
30. T.-N. Phung, “Enhancing the performance of Vietnamese online public service chatbots with RAG,” Lecture Notes in Networks and Systems, vol. 1205, 2024. [Online]. Available: https://doi.org/10.1007/978-3-031-80943-9_8
31. N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang, “Lost in the middle: How language models use long contexts,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 157–173, 2024. [Online]. Available: https://aclanthology.org/2024.tacl-1.9/
32. M. D. Smucker, J. Allan, and B. Carterette, “A comparison of statistical significance tests for information retrieval evaluation,” in Proc. 16th ACM Conference on Information and Knowledge Management (CIKM), 2007, pp. 623–632. [Online]. Available: https://doi.org/10.1145/1321440.1321528
33. Unicode Consortium, “Unicode Standard Annex #15: Unicode normalization forms,” Unicode Technical Report, 2023. [Online]. Available: https://unicode.org/reports/tr15/
34. J. Zobel, “How reliable are the results of large-scale information retrieval experiments?,” in Proc. 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, 1998, pp. 307–314. [Online]. Available: https://doi.org/10.1145/290941.291014