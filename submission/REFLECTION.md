# Reflection — Lab 19

**Tên:** Nguyễn Quốc Tuấn
**Cohort:** 4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

- **Kết quả:** `exact`: BM25 và Hybrid cùng dẫn đầu (96.7% vs Semantic 88.7%) vì từ khóa kỹ thuật khớp nguyên văn. `paraphrase`: BM25 đạt 33.3%, Hybrid 32.0% (bge-small-en hạn chế với tiếng Việt diễn đạt lại; đổi sang bge-m3 sẽ cải thiện). `mixed`: **Hybrid thắng áp đảo tuyệt đối (100.0%)** so với BM25 (97.0%) và Semantic (98.5%) nhờ RRF dung hợp cả 2 nguồn tín hiệu. Chung cuộc Hybrid đạt Precision@10 cao nhất (78.6%).
- **Khi không dùng Hybrid:** Khi bị giới hạn độ trễ và chi phí cực đoan (yêu cầu P99 < 2ms, hàng trăm nghìn QPS không kham nổi chi phí embedding và duy trì 2 index). Chọn **Pure BM25** khi tra cứu mã lỗi, số định danh (SKU, ID), văn bản luật chính xác. Chọn **Pure Vector** cho tìm kiếm đa phương tiện (ảnh, audio) hoặc câu hỏi khái niệm không có từ khóa cố định.

---

## Điều ngạc nhiên nhất khi làm lab này

Hiện tượng *Recall Cliff* khi post-filter sập từ 1.00 về 0.00 chỉ vì bộ lọc chọn lọc 3.8% (buộc over-fetch tới 50% corpus mới bù đắp được), và mức độ rò rỉ target-naive encoding đẩy train AUC lên 0.999 giả tạo.

---

## Bonus challenge

- [ ] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
