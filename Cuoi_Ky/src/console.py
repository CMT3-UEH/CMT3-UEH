# Cho phép in tiếng Việt ra cửa sổ dòng lệnh Windows.

import sys


def setup() -> None:
    # Chuyển stdout và stderr sang UTF-8.
    for luong in (sys.stdout, sys.stderr):
        reconfigure = getattr(luong, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Luồng bị chuyển hướng vào nơi không đổi được mã hoá — bỏ qua,
                # ký tự không hiển thị được sẽ thành dấu hỏi thay vì làm dừng chương trình.
                pass
