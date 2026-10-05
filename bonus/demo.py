#!/usr/bin/env python3
"""
Demo script for Bonus Challenge: HybridMemoryAgent.
Executes 5 distinct retrieval queries illustrating episodic memory + user profile integration.
"""
import sys
from pathlib import Path

# Ensure repo root and bonus are in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bonus"))

from agent import HybridMemoryAgent


def run_demo():
    print("Initializing HybridMemoryAgent...")
    agent = HybridMemoryAgent()

    user_id = "u_001"

    print(f"\nSeeding episodic memories for user '{user_id}'...")
    sample_memories = [
        (
            "Ghi chú Kubernetes: Đã cấu hình Horizontal Pod Autoscaler (HPA) cho cluster production. "
            "Metrics sử dụng CPU 70% và custom latency metric từ Prometheus để trigger scale pods."
        ),
        (
            "Báo cáo Cloud Security: Phân tích sự cố rò rỉ IAM credential tháng trước. "
            "Khuyến nghị áp dụng AWS IAM Roles Anywhere, xoá bỏ long-term access keys và bật AWS GuardDuty."
        ),
        (
            "Kiến trúc Cloud: Thiết kế hệ thống multi-region active-active trên GCP và AWS. "
            "Sử dụng Cloudflare Load Balancing để phân tải Geo-DNS và giảm độ trễ cho người dùng Đông Nam Á."
        ),
        (
            "Tài liệu tự động co giãn: Cơ chế auto-scaling hạ tầng điện toán đám mây cho phép tăng giảm máy ảo "
            "dựa theo biến động lưu lượng truy cập thực tế, tối ưu hóa chi phí đến 40%."
        ),
        (
            "Nghiên cứu AI/ML: Thử nghiệm kỹ thuật RAG với Qdrant vector database và Feast feature store "
            "cho bài toán gợi ý sản phẩm cá nhân hóa theo thời gian thực."
        ),
    ]

    for mem in sample_memories:
        agent.remember(mem, user_id=user_id)

    print(f"Successfully indexed {len(sample_memories)} episodic memory items.")

    queries = [
        (
            "1. Hỏi đơn giản (Vector + Keyword hit)",
            "Tôi đã đọc gì về Kubernetes?",
            "Kỳ vọng: Truy xuất chính xác ghi chú cấu hình HPA và Prometheus metric.",
        ),
        (
            "2. Hỏi cần Profile Context (Topic Affinity)",
            "Recommend đọc gì tiếp dựa trên sở thích của tôi?",
            "Kỳ vọng: Khai thác topic_affinity='cloud' từ Feast profile để boost tài liệu cloud.",
        ),
        (
            "3. Hỏi cần Fresh Activity (Streaming Velocity)",
            "Tôi đang quan tâm gì và hoạt động như thế nào gần đây?",
            "Kỳ vọng: Khai thác queries_last_hour=11 và distinct_topics_24h=4 từ Feast Feature Store.",
        ),
        (
            "4. Hỏi Paraphrase (Vector Semantic Matching)",
            "Tài liệu về tự động mở rộng hạ tầng?",
            "Kỳ vọng: Tìm đúng bài 'auto-scaling hạ tầng' và 'HPA' dù query không chứa từ khóa exact.",
        ),
        (
            "5. Hỏi Mixed (Episodic + Profile + Hybrid Search)",
            "Cho tôi summary cloud security và tối ưu hạ tầng?",
            "Kỳ vọng: RRF kết hợp báo cáo IAM Security và kiến trúc Multi-region theo preferred_language='vi'.",
        ),
    ]

    print("\n" + "=" * 80)
    print("RUNNING 5 DEMONSTRATION QUERIES")
    print("=" * 80)

    for i, (title, q, note) in enumerate(queries, 1):
        print(f"\n>>> QUERY #{i}: {title}")
        print(f"    Intent: {note}")
        context = agent.recall(q, user_id=user_id, top_k=2)
        print(context)

    print("\nDemo finished successfully with exit code 0.")
    return 0


if __name__ == "__main__":
    sys.exit(run_demo())
