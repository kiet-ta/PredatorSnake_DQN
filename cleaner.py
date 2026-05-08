import glob
import os
import time


def clean_garbage(directory, pattern, keep_top_k=None):
    """
    Hàm dọn rác đa năng.
    Nếu có keep_top_k, sẽ giữ lại K file mới nhất.
    Nếu keep_top_k = None, sẽ trảm sạch không chừa file nào.
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)

    if not files:
        return

    # Sắp xếp theo thời gian sửa đổi (Mới nhất đứng đầu)
    files.sort(key=os.path.getmtime, reverse=True)

    if keep_top_k is not None:
        files_to_delete = files[keep_top_k:]
    else:
        files_to_delete = files  # Trảm hết

    for f in files_to_delete:
        try:
            os.remove(f)
            print(f"🗑️ [Daemon] Đã tiêu hủy: {f}")
        except Exception as e:
            print(f"⚠️ [Daemon] Lỗi khi xóa {f}: {e}")


if __name__ == "__main__":
    print("🤖 Khởi động Sidecar Garbage Collector...")
    print("⏳ Tần suất quét: Mỗi 5 phút/lần.")

    while True:
        print("\n--- Bắt đầu chu kỳ dọn rác ---")

        # 1. Quét sạch toàn bộ tàn dư của bộ DQN cũ (Trảm 100%)
        clean_garbage("models/checkpoints", "snake_dqn_*.zip", keep_top_k=None)

        # 2. Dọn dẹp Checkpoint của AlphaZero (Chỉ giữ 3 bản mới nhất)
        # Lưu ý: Sửa lại đường dẫn "alphazero" hoặc "checkpoints" tùy vào chỗ bro đang lưu file .pt
        clean_garbage("models/alphazero", "model_iter_*.pt", keep_top_k=3)

        print("✅ Dọn dẹp hoàn tất. Đi ngủ 5 phút...")

        # Cho luồng ngủ 300 giây (5 phút) rồi mới dậy quét tiếp
        time.sleep(300)
