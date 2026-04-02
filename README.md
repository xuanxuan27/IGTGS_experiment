# IGTGS_experiment

以 **FastAPI** 提供和弦相關工具：從 **91 譜**取得和弦作為 Ground Truth、對 **YouTube** 影片做 **Chord-CNN-LSTM + ChordRefiner** 分析產生 `.lab`，並在網頁上與 GT **對齊比對**（紅字／多餘／漏標等）。

專案根目錄為本 repo 的 **`IGTGS_experiment/`**（內含 `backend/`、`frontend/`、根目錄的 `experiment.py` 等）。

## 功能概覽

| 項目 | 說明 |
|------|------|
| **91 譜** | `POST /api/scrape/91pu` 解析頁面，取得和弦字串列表（需 **Selenium** 與瀏覽器驅動）。 |
| **YouTube** | `GET /api/youtube/search` 以 **yt-dlp** 搜尋；`POST /api/youtube/chord-analyze` 下載音訊、跑雙次辨識（`submission` / `custom` 詞表）、回傳兩份 `.lab` 與暫存音訊 URL。 |
| **比對** | `POST /api/compare/lab`：`.lab` 與 91 譜 GT（含 capo 轉換）對齊、計算多餘／漏標／標錯。 |
| **前端** | `GET /` 導向 `lab_compare.html`：上傳／貼上資料呼叫上述 API。 |

辨識管線實際運算依賴鄰近 clone 的 **IGTGS** 專案之 **`igtgs_backend`**（`beat_service`、`chord_service`、`compat`、Chord-CNN-LSTM 模型與權重）。

## 環境需求
- 安裝 [IGTGS](https://github.com/evankuo2017/IGTGS) 系統
- **Python** 3.10+（建議用 conda／venv）
- **ffmpeg**（YouTube 轉 WAV）
- 系統可執行 **yt-dlp**（`requirements.txt` 已列，或自行安裝）
- **IGTGS**：建議與 `IGTGS_experiment` **同層目錄**放置 `IGTGS/` 或 `igtgs/`（內含 `igtgs_backend/`）；程式會自動解析，或以環境變數指定
- 和弦辨識：**NumPy 須 < 2**（與 **madmom** 二進位相容）；Python 3.10+ 匯入 madmom 前會由根目錄 **`madmom_compat.py`** 處理相容性

## 安裝

```bash
cd IGTGS_experiment
conda create -n IGTGS_experiment python=3.10
conda activate IGTGS_experiment
pip install -r requirements.txt
```

91 譜爬蟲需本機 **Chrome / Chromium** 與對應 **ChromeDriver**（或 Selenium 4 自動管理），並請遵守網站使用條款。

```bash
wget -O /tmp/google-chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt update
sudo apt install -y /tmp/google-chrome.deb
```

若有需要，可安裝

```
sudo apt update
sudo apt install -y libnss3 libatk-bridge2.0-0 libgbm1 libasound2
sudo apt install -y libx11-xcb1 libxcomposite1 libxdamage1 libxrandr2 libgtk-3-0 libxshmfence1
```

## 設定（`.env`）

在 **`IGTGS_experiment` 根目錄**建立 `.env`（與 `experiment.py` 同層）。變數會由 `backend/app/settings.py` 載入（不覆寫已存在之環境變數）。

| 變數 | 說明 |
|------|------|
| `IGTGS_ROOT` | （選用）含 `analysis_engine` 的 IGTGS 專案根；未設時通常使用本 repo 根目錄的精簡腳本。 |
| `IGTGS_BACKEND_DIR` | （選用）直接指向 `igtgs_backend`；未設時會嘗試 `…/IGTGS/igtgs_backend` 等同層路徑。 |
| `AUDIO_SESSION_TTL_SECONDS` | 預設 `3600`；YouTube 分析後暫存音訊之存活秒數。 |
| `AUDIO_SESSIONS_DIR` | （選用）自訂音訊快取目錄；未設則使用 `.IGTGS_experiment_cache/audio_sessions/`。 |

詳見 `backend/app/settings_example.py`。請勿將含私密路徑的 `.env` 提交至版本庫。

## 啟動後端

在 **`IGTGS_experiment` 根目錄**執行（以便讀取 `.env` 與呼叫根目錄 `experiment.py`）：

```bash
cd IGTGS_experiment
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

瀏覽器開啟：

- 介面：<http://127.0.0.1:8000/ui/lab_compare.html>  
- API 文件：<http://127.0.0.1:8000/docs>

## 命令列實驗腳本

根目錄 **`experiment.py`**：對單一音檔跑節拍 + 和弦 + Refiner，並輸出 JSON / `.lab`。

```bash
cd IGTGS_experiment
python experiment.py /path/to/audio.wav -o out.json --lab-out out.lab --chord-dict submission
```

詞表可選：`submission`、`custom`、`full`、`extended`、`ismir2017`（需 IGTGS 模型 `data/` 內對應 `*_chord_list.txt`）。

## API 摘要

- `POST /api/compare/lab` — body：`before_lab`、`after_lab`、`chords_raw`、`gt_capo`、`target_capo`
- `POST /api/scrape/91pu` — body：`url`（91pu 網址）
- `GET /api/youtube/search` — `q`、`max_results`
- `POST /api/youtube/chord-analyze` — body：`url` 或 `video_id`
- `GET /api/audio/session/{session_id}` — 下載先前分析產生之 WAV

## 目錄結構（精簡）

```
IGTGS_experiment/
├── backend/app/          # FastAPI、domain（lab 解析／對齊）、services
├── frontend/             # 靜態頁，如 lab_compare.html
├── experiment.py         # 子程序辨識入口（由 chord-analyze 呼叫）
├── analysis_engine.py    # 呼叫 IGTGS 節拍／和弦服務
├── beat_chord_refinement.py
├── grid_builder.py       # 節拍與和弦時間軸對齊（Refiner 路徑）
├── madmom_compat.py      # madmom / NumPy 版本相容
├── igtgs_paths.py        # 解析 igtgs_backend 路徑
└── requirements.txt
```

## 授權與外部專案

本 repo 可能內嵌或依賴 **IGTGS**、**Chord-CNN-LSTM**、**ChordRefiner** 等外部程式與權重；使用前請遵循各專案授權，並自行備妥模型檔（例如 `cache_data` 下之 `.best`）。
