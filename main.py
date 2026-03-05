"""
Advanced Telegram VC Music Bot (Pyrogram + PyTgCalls + yt-dlp)

Features:
- per-chat persistent queue (sqlite)
- caching with cleanup
- playlist commands (add/list/remove)
- shuffle / repeat modes
- volume re-encode
- admin/owner checks
- robust downloader (threadpool)
- auto cleanup on exit

Important:
- Install system ffmpeg and libopus.
- Keep STRING_SESSION secret.
- Test in private group first.
"""

import os
import asyncio
import shutil
import logging
import uuid
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

from pyrogram import Client, filters
from pyrogram.types import Message
from pyrogram.session import StringSession
from pytgcalls import GroupCallFactory
from yt_dlp import YoutubeDL
import aiofiles
from dotenv import load_dotenv
import aiosqlite
import random
import time
import subprocess

# -------------------- load config --------------------
load_dotenv()
API_ID = int(os.getenv("API_ID") or 0)
API_HASH = os.getenv("API_HASH")
STRING_SESSION = os.getenv("STRING_SESSION")
OWNER_ID = int(os.getenv("OWNER_ID") or 0)
TMP_DIR = Path(os.getenv("TMP_DIR", "./tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MAX = int(os.getenv("CACHE_MAX_FILES", "40"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# logging
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("vc-music")

# yt-dlp options (tweak if needed)
YTDL_OPTS = {
    "format": "bestaudio/best",
    "outtmpl": str(TMP_DIR / "%(id)s.%(ext)s"),
    "noplaylist": False,
    "quiet": True,
    "no_warnings": True,
    "geo_bypass": True,
    "nocheckcertificate": True,
    "ignoreerrors": True,
}

# threadpool for blocking downloads / ffmpeg work
POOL = ThreadPoolExecutor(max_workers=4)

# small helper to run blocking calls
def run_blocking(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(POOL, lambda: fn(*args, **kwargs))


# -------------------- Data models --------------------
@dataclass
class Track:
    id: str               # unique id (yt id or uuid)
    title: str
    filepath: str         # absolute path string
    duration: int         # seconds (0 if unknown)
    url: str
    requested_by: str

    def to_tuple(self, chat_id: int, position: int):
        return (self.id, chat_id, position, self.title, self.filepath, self.duration, self.url, self.requested_by)

# -------------------- DB (sqlite) --------------------
DB_PATH = Path("musicbot.db")

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""CREATE TABLE IF NOT EXISTS queue (
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
        )""")
        await db.execute("""CREATE TABLE IF NOT EXISTS meta (
            chat_id INTEGER PRIMARY KEY,
            repeat_mode TEXT DEFAULT 'none'  -- values: none|one|all
        )""")
        await db.commit()


# -------------------- YTDL helper --------------------
ytdl = YoutubeDL(YTDL_OPTS)

def ytdl_extract_info_blocking(target: str, download: bool = True):
    """Blocking yt-dlp extraction. Returns info dict and filepath (if downloaded)."""
    opts = {**YTDL_OPTS}
    with YoutubeDL(opts) as dl:
        info = dl.extract_info(target, download=download)
        if not info:
            raise Exception("yt-dlp returned no info")
        # handle playlist vs single
        if "entries" in info and isinstance(info["entries"], list):
            # choose first if search
            info = next((e for e in info["entries"] if e), info["entries"][0])
        filepath = None
        if download:
            filepath = Path(dl.prepare_filename(info)).absolute()
        return info, filepath

def ffmpeg_reencode_with_volume(src: Path, dst: Path, volume: float):
    """
    Re-encode audio with ffmpeg applying volume filter.
    volume: 1.0 = original, 0.5 = -6dB, 2.0 = +6dB
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", f"volume={volume}",
        "-c:a", "libopus", "-b:a", "64k",
        str(dst)
    ]
    subprocess.run(cmd, check=True)

# -------------------- MusicBot core --------------------
class MusicBot:
    def __init__(self):
        # Use StringSession (user account) for VC control
        if not STRING_SESSION:
            raise RuntimeError("STRING_SESSION is required in .env")
        self.app = Client(StringSession(STRING_SESSION), api_id=API_ID, api_hash=API_HASH)
        self.gc_factory = None
        self.group_calls: Dict[int, object] = {}
        self.queues_in_memory: Dict[int, List[Track]] = {}  # chat_id -> list of Tracks (cache)
        self.now_playing: Dict[int, Optional[Track]] = {}
        self.repeat_mode: Dict[int, str] = {}  # none|one|all
        self.cache_index: Dict[str, float] = {}  # filepath -> last_used_timestamp
        self.downloader_lock = asyncio.Lock()

        # yt-dlp instance used for metadata-only extraction (non-blocking wrapper uses thread)
        # Register handlers after app ready
        # command handlers use pyrogram decorators below in start()

    async def start(self):
        await init_db()
        await self.app.start()
        self.gc_factory = GroupCallFactory(self.app)
        # attach commands
        self.app.add_handler(self.app.on_message(filters.command("start"))(self.cmd_start))
        self.app.add_handler(self.app.on_message(filters.command("join"))(self.cmd_join))
        self.app.add_handler(self.app.on_message(filters.command("leave"))(self.cmd_leave))
        self.app.add_handler(self.app.on_message(filters.command("play"))(self.cmd_play))
        self.app.add_handler(self.app.on_message(filters.command("stream"))(self.cmd_stream))
        self.app.add_handler(self.app.on_message(filters.command("skip"))(self.cmd_skip))
        self.app.add_handler(self.app.on_message(filters.command("pause"))(self.cmd_pause))
        self.app.add_handler(self.app.on_message(filters.command("resume"))(self.cmd_resume))
        self.app.add_handler(self.app.on_message(filters.command("np"))(self.cmd_nowplaying))
        self.app.add_handler(self.app.on_message(filters.command("queue"))(self.cmd_queue))
        self.app.add_handler(self.app.on_message(filters.command("clear"))(self.cmd_clear))
        self.app.add_handler(self.app.on_message(filters.command("shuffle"))(self.cmd_shuffle))
        self.app.add_handler(self.app.on_message(filters.command("repeat"))(self.cmd_repeat))
        self.app.add_handler(self.app.on_message(filters.command("volume"))(self.cmd_volume))
        self.app.add_handler(self.app.on_message(filters.command("playlist"))(self.cmd_playlist))
        self.app.add_handler(self.app.on_message(filters.command("shutdown") & filters.user(OWNER_ID))(self.cmd_shutdown))

        logger.info("Client started and handlers registered")

    async def stop(self):
        # stop group calls, cleanup
        for gc in list(self.group_calls.values()):
            try:
                await gc.stop()
            except Exception:
                pass
        await self.app.stop()
        # cleanup temp files exceeding cache limit
        await self._enforce_cache_limit()
        logger.info("Stopped")

    # -------------------- Utilities --------------------
    def _is_admin(self, message: Message) -> bool:
        # allow owner OR chat admin
        if message.from_user and message.from_user.id == OWNER_ID:
            return True
        # pyrogram provides chat member queries but sync call here is simple: owner only for now
        return False

    async def _db_enqueue(self, chat_id: int, track: Track):
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT IFNULL(MAX(position), 0) FROM queue WHERE chat_id = ?", (chat_id,))
            row = await cur.fetchone()
            pos = (row[0] or 0) + 1
            await db.execute("""INSERT OR REPLACE INTO queue(track_id, chat_id, position, title, filepath, duration, url, requested_by, added_at)
                                VALUES(?,?,?,?,?,?,?,?,?)""",
                             (track.id, chat_id, pos, track.title, track.filepath, track.duration, track.url, track.requested_by, time.time()))
            await db.commit()
        # update in-memory
        self.queues_in_memory.setdefault(chat_id, []).append(track)
        self.cache_index[str(track.filepath)] = time.time()

    async def _db_pop_next(self, chat_id: int) -> Optional[Track]:
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT track_id, title, filepath, duration, url, requested_by FROM queue WHERE chat_id = ? ORDER BY position ASC LIMIT 1", (chat_id,))
            row = await cur.fetchone()
            if not row:
                return None
            track_id, title, filepath, duration, url, requested_by = row
            # delete that row
            await db.execute("DELETE FROM queue WHERE track_id = ? AND chat_id = ?", (track_id, chat_id))
            # reindex positions (optional)
            await db.commit()
            t = Track(id=track_id, title=title, filepath=filepath, duration=int(duration or 0), url=url, requested_by=requested_by)
            # also pop from in-memory head if present
            if chat_id in self.queues_in_memory and len(self.queues_in_memory[chat_id])>0:
                if self.queues_in_memory[chat_id][0].id == track_id:
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
        self.repeat_mode[chat_id] = mode

    async def _load_repeat(self, chat_id: int) -> str:
        if chat_id in self.repeat_mode:
            return self.repeat_mode[chat_id]
        async with aiosqlite.connect(DB_PATH) as db:
            cur = await db.execute("SELECT repeat_mode FROM meta WHERE chat_id = ?", (chat_id,))
            row = await cur.fetchone()
            mode = row[0] if row and row[0] else "none"
            self.repeat_mode[chat_id] = mode
            return mode

    async def _enforce_cache_limit(self):
        # remove oldest files if count > CACHE_MAX
        files = sorted(self.cache_index.items(), key=lambda x: x[1])  # (path, last_used)
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

    # -------------------- Downloader helpers --------------------
    async def download_track(self, query: str, requested_by: str) -> Track:
        """
        Download a query (URL or search). Returns Track object.
        This runs yt-dlp in a thread to avoid blocking loop.
        """
        # determine target
        is_url = query.startswith("http://") or query.startswith("https://")
        target = query if is_url else f"ytsearch1:{query}"

        # lock to avoid multiple simultaneous downloads for same query (optional)
        async with self.downloader_lock:
            info, filepath = await run_blocking(ytdl.extract_info, target, False), None
            # actually perform download in blocking thread to avoid yt-dlp async complications
            info, filepath = await asyncio.get_event_loop().run_in_executor(POOL, lambda: ytdl.extract_info(target, download=True))
            # handle entries
            if isinstance(info, dict) and "entries" in info and isinstance(info["entries"], list):
                info = info["entries"][0] or info
            # ensure filepath resolution
            filepath = Path(ytdl.prepare_filename(info)).absolute()
            track_id = (info.get("id") or str(uuid.uuid4()))
            title = info.get("title") or "Unknown title"
            duration = int(info.get("duration") or 0)
            url = info.get("webpage_url") or query
            t = Track(id=track_id, title=title, filepath=str(filepath), duration=duration, url=url, requested_by=requested_by)
            # update cache index
            self.cache_index[str(filepath)] = time.time()
            await self._enforce_cache_limit()
            return t

    # -------------------- Playback control --------------------
    async def _ensure_group_call(self, chat_id: int):
        if chat_id in self.group_calls:
            return self.group_calls[chat_id]
        gc = self.gc_factory.get_group_call()
        # on playout ended event -> play next
        @gc.on_playout_ended
        async def _on_end(_, filename):
            logger.info("Playout ended in %s -> %s", chat_id, filename)
            await self._on_track_end(chat_id)
        await gc.start(chat_id)
        self.group_calls[chat_id] = gc
        logger.info("Joined voice chat %s", chat_id)
        return gc

    async def _on_track_end(self, chat_id: int):
        # depending on repeat mode decide what to do
        mode = await self._load_repeat(chat_id)
        current = self.now_playing.get(chat_id)
        if not current:
            # nothing playing
            next_track = await self._db_pop_next(chat_id)
            if next_track:
                await self._start_play(chat_id, next_track)
            return

        if mode == "one":
            # replay same track
            await self._start_play(chat_id, current)
            return
        elif mode == "all":
            # push current to queue tail and play next (round-robin)
            await self._db_enqueue(chat_id, current)
            next_track = await self._db_pop_next(chat_id)
            if next_track:
                await self._start_play(chat_id, next_track)
                return
            else:
                # nothing else -> replay this same
                await self._start_play(chat_id, current)
                return
        else:  # none
            next_track = await self._db_pop_next(chat_id)
            if next_track:
                await self._start_play(chat_id, next_track)
                return
            else:
                self.now_playing[chat_id] = None
                logger.info("Queue empty for chat %s", chat_id)
                # optional: leave after idle timeout

    async def _start_play(self, chat_id: int, track: Track):
        # ensure group call exists
        gc = await self._ensure_group_call(chat_id)
        # ensure file exists (re-download if missing)
        fp = Path(track.filepath)
        if not fp.exists():
            # re-download by URL (blocking), do in thread
            try:
                logger.info("File missing, re-downloading: %s", track.url)
                t2 = await asyncio.get_event_loop().run_in_executor(POOL, lambda: ytdl.extract_info(track.url, download=True))
                fp = Path(ytdl.prepare_filename(t2)).absolute()
                track.filepath = str(fp)
            except Exception as e:
                logger.exception("Failed re-download")
                # skip to next
                await self._on_track_end(chat_id)
                return
        # start audio; pytgcalls start_audio API may vary across versions
        logger.info("Playing in %s: %s (%s)", chat_id, track.title, track.filepath)
        self.now_playing[chat_id] = track
        # start playback (string path)
        await gc.start_audio(str(track.filepath))
        # on_playout_ended handler will trigger after file ends

    # -------------------- Command handlers --------------------
    async def cmd_start(self, client, message: Message):
        await message.reply_text("Advanced VC Music Bot ready.\nCommands: /join /leave /play /stream /skip /pause /resume /np /queue /clear /shuffle /repeat <none|one|all> /volume <0.1-4.0> /playlist add/list/remove")

    async def cmd_join(self, client, message: Message):
        chat_id = message.chat.id
        try:
            await self._ensure_group_call(chat_id)
            await message.reply_text("✅ Joined voice chat.")
        except Exception as e:
            logger.exception("Join error")
            await message.reply_text("Failed to join voice chat: " + str(e))

    async def cmd_leave(self, client, message: Message):
        chat_id = message.chat.id
        if chat_id not in self.group_calls:
            await message.reply_text("Not connected.")
            return
        gc = self.group_calls.pop(chat_id)
        await gc.stop()
        self.now_playing.pop(chat_id, None)
        await message.reply_text("Left voice chat and cleared playing state.")

    async def cmd_play(self, client, message: Message):
        """Downloads track then queues it (reliable)."""
        chat_id = message.chat.id
        if len(message.command) < 2:
            await message.reply_text("Usage: /play <url or search terms>")
            return
        query = " ".join(message.command[1:])
        msg = await message.reply_text("🔎 Searching and downloading...")
        try:
            track = await asyncio.get_event_loop().run_in_executor(POOL, lambda: ytdl.extract_info(query if query.startswith("http") else f"ytsearch1:{query}", download=True))
            # normalize info -> Track
            if "entries" in track and isinstance(track["entries"], list):
                track = track["entries"][0] or track
            filepath = Path(ytdl.prepare_filename(track)).absolute()
            t = Track(id=(track.get("id") or str(uuid.uuid4())),
                      title=track.get("title") or "Unknown",
                      filepath=str(filepath),
                      duration=int(track.get("duration") or 0),
                      url=track.get("webpage_url") or query,
                      requested_by=message.from_user.first_name or str(message.from_user.id))
            # enqueue
            await self._db_enqueue(chat_id, t)
            await msg.edit_text(f"Queued: {t.title}")
            # if nothing is playing -> start next
            if self.now_playing.get(chat_id) is None:
                next_t = await self._db_pop_next(chat_id)
                if next_t:
                    await self._start_play(chat_id, next_t)
        except Exception as e:
            logger.exception("Play error")
            await msg.edit_text("Failed to fetch/play: " + str(e))

    async def cmd_stream(self, client, message: Message):
        """Attempt to stream via URL without full download. Works for some sources."""
        chat_id = message.chat.id
        if len(message.command) < 2:
            await message.reply_text("Usage: /stream <direct-audio-url or youtube link>")
            return
        query = " ".join(message.command[1:])
        msg = await message.reply_text("🔎 Preparing stream...")
        try:
            # yt-dlp get best direct audio URL (no download)
            info = await asyncio.get_event_loop().run_in_executor(POOL, lambda: ytdl.extract_info(query if query.startswith("http") else f"ytsearch1:{query}", download=False))
            if "entries" in info:
                info = info["entries"][0] or info
            # best direct url in formats
            formats = info.get("formats") or []
            # pick audio format with url
            audio_url = None
            for f in reversed(formats):
                if f.get("acodec") != "none" and f.get("url"):
                    audio_url = f.get("url")
                    break
            if not audio_url:
                await msg.edit_text("No streamable audio format found; try /play to download.")
                return
            # create a temp small file wrapper that will be played by pytgcalls (some versions accept HTTP URLs)
            # Many pytgcalls versions accept direct URLs; try start_audio(url)
            gc = await self._ensure_group_call(chat_id)
            await gc.start_audio(audio_url)
            await msg.edit_text("Streaming started (best-effort).")
        except Exception as e:
            logger.exception("Stream error")
            await msg.edit_text("Failed to stream: " + str(e))

    async def cmd_skip(self, client, message: Message):
        chat_id = message.chat.id
        if chat_id not in self.group_calls:
            await message.reply_text("Not in voice chat.")
            return
        gc = self.group_calls[chat_id]
        try:
            await gc.stop_playout()
            await message.reply_text("⏭ Skipped current.")
        except Exception as e:
            logger.exception("Skip error")
            await message.reply_text("Failed to skip: " + str(e))

    async def cmd_pause(self, client, message: Message):
        chat_id = message.chat.id
        gc = self.group_calls.get(chat_id)
        if not gc:
            await message.reply_text("Not in voice chat.")
            return
        try:
            await gc.pause_playout()
            await message.reply_text("⏸ Paused.")
        except Exception as e:
            logger.exception("Pause error")
            await message.reply_text("Failed to pause: " + str(e))

    async def cmd_resume(self, client, message: Message):
        chat_id = message.chat.id
        gc = self.group_calls.get(chat_id)
        if not gc:
            await message.reply_text("Not in voice chat.")
            return
        try:
            await gc.resume_playout()
            await message.reply_text("▶ Resumed.")
        except Exception as e:
            logger.exception("Resume error")
            await message.reply_text("Failed to resume: " + str(e))

    async def cmd_nowplaying(self, client, message: Message):
        chat_id = message.chat.id
        t = self.now_playing.get(chat_id)
        if not t:
            await message.reply_text("Nothing is playing.")
            return
        await message.reply_text(f"Now playing: {t.title}\nRequested by: {t.requested_by}\nDuration: {t.duration}s")

    async def cmd_queue(self, client, message: Message):
        chat_id = message.chat.id
        q = await self._db_get_queue(chat_id)
        if not q:
            await message.reply_text("Queue is empty.")
            return
        text = "Upcoming:\n" + "\n".join([f"{i+1}. {track.title} ({track.duration}s) — {track.requested_by}" for i, track in enumerate(q[:20])])
        await message.reply_text(text)

    async def cmd_clear(self, client, message: Message):
        chat_id = message.chat.id
        if not self._is_admin(message):
            await message.reply_text("Only owner/admin can clear queue.")
            return
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM queue WHERE chat_id = ?", (chat_id,))
            await db.commit()
        self.queues_in_memory.pop(chat_id, None)
        await message.reply_text("Cleared queue.")

    async def cmd_shuffle(self, client, message: Message):
        chat_id = message.chat.id
        if not self._is_admin(message):
            await message.reply_text("Only owner/admin can shuffle.")
            return
        q = await self._db_get_queue(chat_id)
        random.shuffle(q)
        # write back shuffled order
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM queue WHERE chat_id = ?", (chat_id,))
            for pos, t in enumerate(q, start=1):
                await db.execute("""INSERT INTO queue(track_id, chat_id, position, title, filepath, duration, url, requested_by, added_at)
                                    VALUES(?,?,?,?,?,?,?,?,?)""", (t.id, chat_id, pos, t.title, t.filepath, t.duration, t.url, t.requested_by, time.time()))
            await db.commit()
        await message.reply_text("Shuffled queue.")

    async def cmd_repeat(self, client, message: Message):
        chat_id = message.chat.id
        if len(message.command) < 2:
            mode = await self._load_repeat(chat_id)
            await message.reply_text(f"Current repeat mode: {mode}")
            return
        mode = message.command[1].lower()
        if mode not in ("none", "one", "all"):
            await message.reply_text("Usage: /repeat <none|one|all>")
            return
        await self._save_repeat(chat_id, mode)
        await message.reply_text(f"Repeat mode set to {mode}")

    async def cmd_volume(self, client, message: Message):
        """Re-encode current track with volume and restart playback."""
        chat_id = message.chat.id
        if len(message.command) < 2:
            await message.reply_text("Usage: /volume <0.1-4.0>")
            return
        try:
            vol = float(message.command[1])
            if not (0.1 <= vol <= 4.0):
                raise ValueError
        except:
            await message.reply_text("Volume must be a number between 0.1 and 4.0")
            return
        t = self.now_playing.get(chat_id)
        if not t:
            await message.reply_text("Nothing is playing.")
            return
        src = Path(t.filepath)
        if not src.exists():
            await message.reply_text("Source file missing.")
            return
        tmp_dst = TMP_DIR / f"vol_{uuid.uuid4().hex}.ogg"
        await message.reply_text("Re-encoding with new volume (this may take a few seconds)...")
        try:
            # run ffmpeg re-encode in thread
            await asyncio.get_event_loop().run_in_executor(POOL, ffmpeg_reencode_with_volume, src, tmp_dst, vol)
            # replace now playing with new file
            t.filepath = str(tmp_dst)
            # restart playback
            gc = self.group_calls.get(chat_id)
            if not gc:
                await self._ensure_group_call(chat_id)
                gc = self.group_calls.get(chat_id)
            await gc.stop_playout()
            await self._start_play(chat_id, t)
            await message.reply_text("Volume changed.")
        except Exception as e:
            logger.exception("Volume change failed")
            await message.reply_text("Failed to change volume: " + str(e))

    async def cmd_playlist(self, client, message: Message):
        """
        Usage:
        /playlist add <url or search>
        /playlist list
        /playlist remove <index>
        """
        chat_id = message.chat.id
        if len(message.command) < 2:
            await message.reply_text("Usage: /playlist add/list/remove")
            return
        sub = message.command[1].lower()
        if sub == "add":
            if len(message.command) < 3:
                await message.reply_text("Usage: /playlist add <url or search>")
                return
            query = " ".join(message.command[2:])
            msg = await message.reply_text("Adding to playlist (download)...")
            try:
                # simply reuse play logic: download + db enqueue
                track = await asyncio.get_event_loop().run_in_executor(POOL, lambda: ytdl.extract_info(query if query.startswith("http") else f"ytsearch1:{query}", download=True))
                if "entries" in track and isinstance(track["entries"], list):
                    track = track["entries"][0] or track
                filepath = Path(ytdl.prepare_filename(track)).absolute()
                t = Track(id=(track.get("id") or str(uuid.uuid4())),
                          title=track.get("title") or "Unknown",
                          filepath=str(filepath),
                          duration=int(track.get("duration") or 0),
                          url=track.get("webpage_url") or query,
                          requested_by=message.from_user.first_name or str(message.from_user.id))
                await self._db_enqueue(chat_id, t)
                await msg.edit_text(f"Added: {t.title}")
            except Exception as e:
                logger.exception("Playlist add failed")
                await msg.edit_text("Failed to add: " + str(e))
        elif sub == "list":
            q = await self._db_get_queue(chat_id)
            if not q:
                await message.reply_text("Playlist is empty.")
                return
            text = "Playlist:\n" + "\n".join([f"{i+1}. {tr.title} — {tr.requested_by}" for i, tr in enumerate(q[:50])])
            await message.reply_text(text)
        elif sub == "remove":
            if len(message.command) < 3:
                await message.reply_text("Usage: /playlist remove <index>")
                return
            try:
                idx = int(message.command[2]) - 1
                q = await self._db_get_queue(chat_id)
                if idx < 0 or idx >= len(q):
                    await message.reply_text("Index out of range.")
                    return
                rem = q[idx]
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute("DELETE FROM queue WHERE track_id = ? AND chat_id = ?", (rem.id, chat_id))
                    # reindex positions: simple approach - delete and reinsert others
                    rest = [x for i,x in enumerate(q) if i != idx]
                    await db.execute("DELETE FROM queue WHERE chat_id = ?", (chat_id,))
                    for pos, t in enumerate(rest, start=1):
                        await db.execute("""INSERT INTO queue(track_id, chat_id, position, title, filepath, duration, url, requested_by, added_at)
                                            VALUES(?,?,?,?,?,?,?,?,?)""", (t.id, chat_id, pos, t.title, t.filepath, t.duration, t.url, t.requested_by, time.time()))
                    await db.commit()
                await message.reply_text(f"Removed: {rem.title}")
            except Exception as e:
                logger.exception("Playlist remove failed")
                await message.reply_text("Failed to remove: " + str(e))
        else:
            await message.reply_text("Usage: /playlist add/list/remove")

    async def cmd_shutdown(self, client, message: Message):
        await message.reply_text("Owner requested shutdown. Stopping...")
        await self.stop()
        os._exit(0)


# -------------------- Run --------------------
if __name__ == "__main__":
    bot = MusicBot()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.start())
        logger.info("Bot started — idle.")
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopping...")
        loop.run_until_complete(bot.stop())
