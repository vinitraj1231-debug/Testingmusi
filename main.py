# main.py
"""
Robust modern Pyrogram v2 + PyTgCalls assistant (advanced features).
This file is written to tolerate multiple pytgcalls versions by probing
available join/play methods at runtime.
"""

import os
import asyncio
import logging
import uuid
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from pyrogram import Client, filters, idle
from pyrogram.types import Message
from yt_dlp import YoutubeDL
import aiosqlite
import random

# guarded pytgcalls imports
try:
    from pytgcalls import PyTgCalls
    # AudioPiped may be at different paths depending on version
    try:
        from pytgcalls.types.input_stream import AudioPiped
        AUDIO_PIPED_AVAILABLE = True
    except Exception:
        AudioPiped = None
        AUDIO_PIPED_AVAILABLE = False
except Exception as e:
    raise RuntimeError("pytgcalls is not installed or incompatible. Install with: pip install --upgrade pytgcalls") from e

# ---------------- config ----------------
load_dotenv()
API_ID = int(os.getenv("API_ID") or 0)
API_HASH = os.getenv("API_HASH") or ""
STRING_SESSION = os.getenv("STRING_SESSION") or ""
OWNER_ID = int(os.getenv("OWNER_ID") or 0)
TMP_DIR = Path(os.getenv("TMP_DIR", "./tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MAX = int(os.getenv("CACHE_MAX_FILES", "60"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("vc-music")

YTDL_OPTS = {
    "format": "bestaudio/best",
    "outtmpl": str(TMP_DIR / "%(id)s.%(ext)s"),
    "quiet": True,
    "no_warnings": True,
    "noplaylist": False,
    "ignoreerrors": True
}
ytdl = YoutubeDL(YTDL_OPTS)
POOL = ThreadPoolExecutor(max_workers=4)
DB_PATH = Path("musicbot.db")

# ---------------- models ----------------
@dataclass
class Track:
    id: str
    title: str
    filepath: str
    duration: int
    url: str
    requested_by: str

# ---------------- DB ----------------
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS queue (
                track_id TEXT,
                chat_id INTEGER,
                position INTEGER,
                title TEXT,
                filepath TEXT,
                duration INTEGER,
                url TEXT,
                requested_by TEXT,
                added_at REAL,
                PRIMARY KEY(track_id, chat_id)
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                chat_id INTEGER PRIMARY KEY,
                repeat_mode TEXT DEFAULT 'none'
            )
        """)
        await db.commit()

# ---------------- helpers ----------------
def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(POOL, lambda: fn(*args, **kwargs))

def ffmpeg_reencode_with_volume(src: Path, dst: Path, volume: float):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", f"volume={volume}",
        "-c:a", "libopus", "-b:a", "64k",
        str(dst)
    ]
    subprocess.run(cmd, check=True)

# ---------------- PyTgCalls compatibility wrapper ----------------
async def safe_join_call(pytgcalls_obj: Any, chat_id: int, source: Any) -> None:
    """
    Try multiple join/play method names & calling conventions until one works.
    source may be: AudioPiped(...) instance OR file path string OR direct audio URL.
    """
    # candidate method names and likely argument orders
    candidates: List[Dict[str, Any]] = [
        {"name": "join_group_call", "order": ("chat_id", "source")},
        {"name": "join_call", "order": ("chat_id", "source")},
        {"name": "start_group_call", "order": ("chat_id", "source")},
        {"name": "start_call", "order": ("chat_id", "source")},
        {"name": "join_group_call", "order": ("source", "chat_id")},  # some older libs reverse params
        {"name": "join", "order": ("chat_id", "source")},
        {"name": "join", "order": ("source", "chat_id")},
        {"name": "play", "order": ("chat_id", "source")},
        {"name": "play", "order": ("source", "chat_id")},
        {"name": "start", "order": ("chat_id", "source")},
    ]

    last_exc: Optional[Exception] = None
    for cand in candidates:
        func = getattr(pytgcalls_obj, cand["name"], None)
        if not callable(func):
            continue
        try:
            # prepare args according to order
            if cand["order"] == ("chat_id", "source"):
                res = func(chat_id, source)
            else:
                res = func(source, chat_id)
            # if returns coroutine, await it
            if asyncio.iscoroutine(res):
                await res
            return
        except TypeError as te:
            # wrong signature, try next candidate
            last_exc = te
            logger.debug("pytgcalls candidate %s raised TypeError: %s", cand["name"], te)
            continue
        except Exception as e:
            # some implement method but raise other runtime errors (permission, etc.) — bubble that up
            logger.exception("pytgcalls candidate %s raised:", cand["name"])
            last_exc = e
            # if it's fatal, rethrow
            raise
    # if we exhausted all options
    raise RuntimeError("No compatible pytgcalls join/play method found. Last error: %s" % (last_exc,))

# ---------------- Assistant class (modern, robust) ----------------
class MusicAssistant:
    def __init__(self):
        if not STRING_SESSION:
            raise RuntimeError("STRING_SESSION required in .env — generate with gen.py or python3 -m pyrogram")
        self.app = Client(name="vc_assistant", session_string=STRING_SESSION, api_id=API_ID, api_hash=API_HASH)
        self.pytgcalls = PyTgCalls(self.app)
        self.queues_in_memory: Dict[int, List[Track]] = {}
        self.now_playing: Dict[int, Optional[Track]] = {}
        self.play_tasks: Dict[int, asyncio.Task] = {}
        self.cache_index: Dict[str, float] = {}
        self.downloader_lock = asyncio.Lock()

    # DB helpers (enqueue/pop/list)
    async def _db_enqueue(self, chat_id: int, track: Track):
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT IFNULL(MAX(position), 0) FROM queue WHERE chat_id = ?", (chat_id,))
            row = await cur.fetchone()
            pos = (row[0] or 0) + 1
            await db.execute("""INSERT OR REPLACE INTO queue(track_id, chat_id, position, title, filepath, duration, url, requested_by, added_at)
                        VALUES(?,?,?,?,?,?,?,?,?)""",
                             (track.id, chat_id, pos, track.title, track.filepath, track.duration, track.url, track.requested_by, time.time()))
            await db.commit()
        self.queues_in_memory.setdefault(chat_id, []).append(track)
        self.cache_index[str(track.filepath)] = time.time()

    async def _db_pop_next(self, chat_id: int) -> Optional[Track]:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT track_id, title, filepath, duration, url, requested_by FROM queue WHERE chat_id = ? ORDER BY position ASC LIMIT 1", (chat_id,))
            row = await cur.fetchone()
            if not row:
                return None
            track_id, title, filepath, duration, url, requested_by = row
            await db.execute("DELETE FROM queue WHERE track_id = ? AND chat_id = ?", (track_id, chat_id))
            await db.commit()
            t = Track(id=track_id, title=title, filepath=filepath, duration=int(duration or 0), url=url, requested_by=requested_by)
            if chat_id in self.queues_in_memory and self.queues_in_memory[chat_id] and self.queues_in_memory[chat_id][0].id == track_id:
                self.queues_in_memory[chat_id].pop(0)
            return t

    async def _db_get_queue(self, chat_id: int) -> List[Track]:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT track_id, title, filepath, duration, url, requested_by FROM queue WHERE chat_id = ? ORDER BY position ASC", (chat_id,))
            rows = await cur.fetchall()
            return [Track(id=r[0], title=r[1], filepath=r[2], duration=int(r[3] or 0), url=r[4], requested_by=r[5]) for r in rows]

    async def _save_repeat(self, chat_id: int, mode: str):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("INSERT OR REPLACE INTO meta(chat_id, repeat_mode) VALUES(?,?)", (chat_id, mode))
            await db.commit()

    async def _load_repeat(self, chat_id: int) -> str:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT repeat_mode FROM meta WHERE chat_id = ?", (chat_id,))
            row = await cur.fetchone()
            return row[0] if row and row[0] else "none"

    async def _enforce_cache_limit(self):
        files = sorted(self.cache_index.items(), key=lambda x: x[1])
        if len(files) <= CACHE_MAX:
            return
        remove_count = len(files) - CACHE_MAX
        for i in range(remove_count):
            p = Path(files[i][0])
            try:
                p.unlink(missing_ok=True)
                logger.info("Cache removed: %s", p)
                del self.cache_index[str(p)]
            except Exception as e:
                logger.warning("Failed to remove cache %s: %s", p, e)

    # yt-dlp download helper
    async def download_track(self, query: str, requested_by: str) -> Track:
        is_url = query.startswith("http://") or query.startswith("https://")
        target = query if is_url else f"ytsearch1:{query}"
        async with self.downloader_lock:
            info = await run_blocking(ytdl.extract_info, target, False)
            if isinstance(info, dict) and "entries" in info:
                info = info["entries"][0] or info
            info = await run_blocking(ytdl.extract_info, target, True)
            if isinstance(info, dict) and "entries" in info:
                info = info["entries"][0] or info
            filepath = Path(ytdl.prepare_filename(info)).absolute()
            track_id = info.get("id") or str(uuid.uuid4())
            title = info.get("title") or "Unknown"
            duration = int(info.get("duration") or 0)
            url = info.get("webpage_url") or query
            t = Track(id=track_id, title=title, filepath=str(filepath), duration=duration, url=url, requested_by=requested_by)
            self.cache_index[str(filepath)] = time.time()
            await self._enforce_cache_limit()
            return t

    # play / fallback logic using safe_join_call
    async def _start_play(self, chat_id: int, track: Track):
        fp = Path(track.filepath)
        if not fp.exists():
            try:
                t2 = await self.download_track(track.url, track.requested_by)
                fp = Path(t2.filepath)
                track.filepath = str(fp)
            except Exception as e:
                logger.exception("re-download failed")
                await self._play_next(chat_id)
                return

        # pick preferred source
        # Try AudioPiped if available
        source = None
        if AUDIO_PIPED_AVAILABLE and 'AudioPiped' in globals() and AudioPiped is not None:
            try:
                source = AudioPiped(str(fp))
            except Exception:
                source = None

        # try join using safe wrapper; it will try multiple styles
        try:
            # ensure pytgcalls started
            try:
                await self.pytgcalls.start()
            except Exception:
                # some versions require .start() called earlier; ignore if already started
                pass

            if source is not None:
                await safe_join_call(self.pytgcalls, chat_id, source)
            else:
                # first try passing file path
                try:
                    await safe_join_call(self.pytgcalls, chat_id, str(fp))
                except Exception:
                    # final fallback: find direct audio url via yt-dlp and pass that
                    info = await run_blocking(ytdl.extract_info, track.url, False)
                    if isinstance(info, dict) and "entries" in info:
                        info = info["entries"][0] or info
                    audio_url = None
                    formats = info.get("formats") or []
                    for f in reversed(formats):
                        if f.get("acodec") != "none" and f.get("url"):
                            audio_url = f.get("url")
                            break
                    if audio_url:
                        await safe_join_call(self.pytgcalls, chat_id, audio_url)
                    else:
                        raise RuntimeError("No streamable fallback (file, AudioPiped, or direct URL) found.")
        except Exception as exc:
            logger.exception("playback start failed")
            raise RuntimeError(f"Playback failed: {exc}") from exc

        self.now_playing[chat_id] = track

        # monitor end using duration
        if chat_id in self.play_tasks and not self.play_tasks[chat_id].done():
            self.play_tasks[chat_id].cancel()

        async def monitor():
            try:
                wait = max(1, track.duration)
                await asyncio.sleep(wait + 1)
                await self._play_next(chat_id)
            except asyncio.CancelledError:
                return

        self.play_tasks[chat_id] = asyncio.create_task(monitor())

    async def _play_next(self, chat_id: int):
        next_track = await self._db_pop_next(chat_id)
        if next_track:
            await self._start_play(chat_id, next_track)
        else:
            self.now_playing.pop(chat_id, None)
            # optionally leave after idle
            async def idle_leave():
                await asyncio.sleep(20)
                if self.now_playing.get(chat_id) is None:
                    try:
                        await self.pytgcalls.leave_group_call(chat_id)
                    except Exception:
                        pass
            asyncio.create_task(idle_leave())

    # ---- commands ---
    async def cmd_join(self, client: Client, message: Message):
        await message.reply_text("Assistant ready. Use /play <url or search> to queue songs.")
        try:
            await self.pytgcalls.start()
        except Exception:
            pass

    async def cmd_play(self, client: Client, message: Message):
        chat_id = message.chat.id
        if len(message.command) < 2:
            await message.reply_text("Usage: /play <youtube link or search>")
            return
        query = " ".join(message.command[1:])
        msg = await message.reply_text("🔎 Searching & downloading...")
        try:
            t = await self.download_track(query, message.from_user.first_name or str(message.from_user.id))
            await self._db_enqueue(chat_id, t)
            await msg.edit_text(f"Queued: {t.title} — {t.duration}s")
            if not self.now_playing.get(chat_id):
                next_t = await self._db_pop_next(chat_id)
                if next_t:
                    await self._start_play(chat_id, next_t)
        except Exception as e:
            logger.exception("play error")
            await msg.edit_text("Failed to fetch/play: " + str(e))

    async def cmd_skip(self, client: Client, message: Message):
        chat_id = message.chat.id
        if chat_id in self.play_tasks:
            self.play_tasks[chat_id].cancel()
        # best-effort stop/leave action
        try:
            # try leave_group_call if present
            func = getattr(self.pytgcalls, "leave_group_call", None) or getattr(self.pytgcalls, "leave", None)
            if callable(func):
                res = func(chat_id) if func.__code__.co_argcount > 1 else func()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass
        await self._play_next(chat_id)
        await message.reply_text("⏭ Skipped.")

    async def cmd_leave(self, client: Client, message: Message):
        chat_id = message.chat.id
        try:
            func = getattr(self.pytgcalls, "leave_group_call", None) or getattr(self.pytgcalls, "leave", None)
            if callable(func):
                res = func(chat_id) if func.__code__.co_argcount > 1 else func()
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            pass
        self.now_playing.pop(chat_id, None)
        await message.reply_text("Left voice chat.")

    async def cmd_np(self, client: Client, message: Message):
        t = self.now_playing.get(message.chat.id)
        if not t:
            await message.reply_text("Nothing is playing.")
            return
        await message.reply_text(f"Now playing: {t.title}\nRequested by: {t.requested_by}\nDuration: {t.duration}s")

    async def cmd_queue(self, client: Client, message: Message):
        q = await self._db_get_queue(message.chat.id)
        if not q:
            await message.reply_text("Queue empty.")
            return
        txt = "Upcoming:\n" + "\n".join([f"{i+1}. {tr.title} ({tr.duration}s) — {tr.requested_by}" for i, tr in enumerate(q[:20])])
        await message.reply_text(txt)

    async def cmd_clear(self, client: Client, message: Message):
        if message.from_user and message.from_user.id != OWNER_ID:
            await message.reply_text("Only owner can clear.")
            return
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM queue WHERE chat_id = ?", (message.chat.id,))
            await db.commit()
        await message.reply_text("Cleared queue.")

    async def cmd_shuffle(self, client: Client, message: Message):
        if message.from_user and message.from_user.id != OWNER_ID:
            await message.reply_text("Only owner can shuffle.")
            return
        q = await self._db_get_queue(message.chat.id)
        random.shuffle(q)
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM queue WHERE chat_id = ?", (message.chat.id,))
            for pos, t in enumerate(q, start=1):
                await db.execute("""INSERT INTO queue(track_id, chat_id, position, title, filepath, duration, url, requested_by, added_at)
                                    VALUES(?,?,?,?,?,?,?,?,?)""", (t.id, message.chat.id, pos, t.title, t.filepath, t.duration, t.url, t.requested_by, time.time()))
            await db.commit()
        await message.reply_text("Shuffled queue.")

    async def cmd_repeat(self, client: Client, message: Message):
        if len(message.command) < 2:
            mode = await self._load_repeat(message.chat.id)
            await message.reply_text(f"Repeat: {mode}")
            return
        mode = message.command[1].lower()
        if mode not in ("none", "one", "all"):
            await message.reply_text("Usage: /repeat <none|one|all>")
            return
        await self._save_repeat(message.chat.id, mode)
        await message.reply_text(f"Repeat set to {mode}")

    async def cmd_shutdown(self, client: Client, message: Message):
        if message.from_user and message.from_user.id != OWNER_ID:
            await message.reply_text("Only owner can shutdown.")
            return
        await message.reply_text("Shutting down...")
        try:
            await self.pytgcalls.stop()
            await self.app.stop()
        finally:
            os._exit(0)

    # register & start
    async def start(self):
        await init_db()
        await self.app.start()
        self.app.add_handler(self.app.on_message(filters.command("join") & filters.group)(self.cmd_join))
        self.app.add_handler(self.app.on_message(filters.command("play") & filters.group)(self.cmd_play))
        self.app.add_handler(self.app.on_message(filters.command("skip") & filters.group)(self.cmd_skip))
        self.app.add_handler(self.app.on_message(filters.command("leave") & filters.group)(self.cmd_leave))
        self.app.add_handler(self.app.on_message(filters.command("np") & filters.group)(self.cmd_np))
        self.app.add_handler(self.app.on_message(filters.command("queue") & filters.group)(self.cmd_queue))
        self.app.add_handler(self.app.on_message(filters.command("clear") & filters.group)(self.cmd_clear))
        self.app.add_handler(self.app.on_message(filters.command("shuffle") & filters.group)(self.cmd_shuffle))
        self.app.add_handler(self.app.on_message(filters.command("repeat") & filters.group)(self.cmd_repeat))
        self.app.add_handler(self.app.on_message(filters.command("shutdown") & filters.user(OWNER_ID))(self.cmd_shutdown))

        # try safe start
        try:
            await self.pytgcalls.start()
        except Exception:
            pass
        logger.info("Assistant started")

    async def stop(self):
        try:
            await self.pytgcalls.stop()
        except Exception:
            pass
        await self.app.stop()
        await self._enforce_cache_limit()

# ---------- run ----------
async def main():
    assistant = MusicAssistant()
    await assistant.start()
    print("Music assistant running. Use /play in group to queue songs.")
    await idle()
    await assistant.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping...")
