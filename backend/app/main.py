import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .audio_session_store import cleanup_expired, get_session_path, register_session
from .domain.align_compare import run_compare
from .services.chord_experiment_runner import run_experiment_lab
from .services.scrape_91pu import scrape_and_parse
from .services.youtube_audio import download_youtube_audio
from .services.youtube_search import youtube_search_videos
from .settings import get_settings

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UI_DIR = _REPO_ROOT / "frontend"


def _audio_sessions_base() -> Path:
    s = get_settings()
    if s.audio_sessions_dir.strip():
        return Path(s.audio_sessions_dir).expanduser().resolve()
    return _REPO_ROOT / ".chord_ui_cache" / "audio_sessions"


async def _cleanup_loop() -> None:
    while True:
        await asyncio.sleep(300)
        n = cleanup_expired()
        if n:
            _log.debug("已清理過期音訊 session：%s 個", n)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        cleanup_expired()


app = FastAPI(title="chord_ui", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Scrape91PuBody(BaseModel):
    url: str = Field(..., min_length=8, description="91 譜完整頁面 URL")


class CompareLabBody(BaseModel):
    before_lab: str = ""
    after_lab: str = ""
    chords_raw: list[str] = Field(default_factory=list)
    gt_capo: int = Field(0, ge=0, le=8)
    target_capo: int = Field(0, ge=0, le=8)


class YouTubeAnalyzeBody(BaseModel):
    video_id: str | None = None
    url: str | None = None


@app.post("/api/compare/lab")
def api_compare_lab(body: CompareLabBody):
    return run_compare(
        body.before_lab,
        body.after_lab,
        body.chords_raw,
        body.gt_capo,
        body.target_capo,
    )


@app.post("/api/scrape/91pu")
def api_scrape_91pu(body: Scrape91PuBody):
    url = body.url.strip()
    if "91pu.com.tw" not in url and "91pu.com" not in url:
        raise HTTPException(
            status_code=400,
            detail="請提供 91 譜網址（需包含 91pu 網域）",
        )
    try:
        chords = scrape_and_parse(url)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"爬蟲或解析失敗：{e!s}",
        ) from e
    return {"source": url, "chords": chords}


@app.get("/api/youtube/search")
def api_youtube_search(
    q: str = Query(..., min_length=1, description="搜尋關鍵字"),
    max_results: int = Query(10, ge=1, le=25),
):
    try:
        items = youtube_search_videos(q, max_results)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"query": q, "items": items, "search_backend": "yt-dlp"}


@app.post("/api/youtube/chord-analyze")
def api_youtube_chord_analyze(body: YouTubeAnalyzeBody):
    link: str | None = None
    if body.url and body.url.strip():
        link = body.url.strip()
    elif body.video_id and body.video_id.strip():
        link = f"https://www.youtube.com/watch?v={body.video_id.strip()}"
    if not link:
        raise HTTPException(status_code=400, detail="請提供 url 或 video_id")

    settings = get_settings()
    cleanup_expired()

    work = Path(mkdtemp(prefix="yt_chord_"))
    try:
        wav = download_youtube_audio(link, work)
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=502, detail=f"下載音訊失敗：{e}") from e

    ig = settings.igtgs_root.strip() or None
    try:
        before_lab, before_meta = run_experiment_lab(wav, "submission", ig)
        after_lab, after_meta = run_experiment_lab(wav, "custom", ig)
    except Exception as e:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(
            status_code=502,
            detail=f"和弦分析失敗（需 IGTGS 與 experiment.py，並設定 IGTGS_ROOT）：{e}",
        ) from e

    sid = uuid4().hex
    cache_dir = _audio_sessions_base()
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{sid}.wav"

    try:
        shutil.move(str(wav.resolve()), str(dest))
    except OSError:
        shutil.copy2(str(wav.resolve()), str(dest))
        if wav.is_file():
            try:
                wav.unlink()
            except OSError:
                pass

    shutil.rmtree(work, ignore_errors=True)
    register_session(sid, dest, settings.audio_session_ttl_seconds)

    return {
        "before_lab": before_lab,
        "after_lab": after_lab,
        "before_meta": {
            "segmentCount": before_meta.get("segmentCount"),
            "refinedCount": before_meta.get("refinedCount"),
            "chordDict": before_meta.get("chordDict"),
        },
        "after_meta": {
            "segmentCount": after_meta.get("segmentCount"),
            "refinedCount": after_meta.get("refinedCount"),
            "chordDict": after_meta.get("chordDict"),
        },
        "audio_session_id": sid,
        "audio_url": f"/api/audio/session/{sid}",
        "audio_expires_in_seconds": settings.audio_session_ttl_seconds,
    }


@app.get("/api/audio/session/{session_id}")
def api_audio_session(session_id: str):
    p = get_session_path(session_id)
    if not p or not p.is_file():
        raise HTTPException(status_code=404, detail="音訊已失效或不存在")
    return FileResponse(
        path=p,
        media_type="audio/wav",
        filename="youtube_audio.wav",
    )


@app.get("/")
def root():
    return RedirectResponse(url="/ui/lab_compare.html", status_code=302)


app.mount(
    "/ui",
    StaticFiles(directory=str(_UI_DIR), html=False),
    name="ui",
)
