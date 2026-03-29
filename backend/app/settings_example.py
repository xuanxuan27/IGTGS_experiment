"""
（範例說明，非執行用模組）

專案根目錄為 **chord_ui**（含 backend/、frontend/、experiment.py）；`.env` 放在此層。

和弦分析需要 IGTGS 的 `igtgs_backend`（beat_service、chord_service、compat、models）。

chord_ui 根目錄可只放精簡腳本；後端目錄會自動解析為與 chord_ui 同層的 `IGTGS/igtgs_backend`（或 `igtgs/`）。

可選方式：
1. `.env`：`IGTGS_ROOT`（給 experiment 模組路徑）、必要時 `IGTGS_BACKEND_DIR=/path/to/igtgs_backend`
2. 將 IGTGS clone 在與 chord_ui 同層的 `igtgs/` 或 `IGTGS/`
3. 或整包放在 chord_ui 底下的 `igtgs/` 子目錄

其他選用變數：

    AUDIO_SESSION_TTL_SECONDS=3600
    AUDIO_SESSIONS_DIR=
"""
