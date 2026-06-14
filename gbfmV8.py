#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  🐾  Geeky Beast Radio  —  v8  (GBRadio)                                   ║
# ║  Global FM Station Finder · Live DSP · Arctic Edition · AI Intelligence     ║
# ║                                                                              ║
# ║  UPGRADED in v8 (Speech Lab / Calibration Recorder):                       ║
# ║  • SpeechCalibrationRecorder — live FM audio → WAV dataset engine          ║
# ║    R key reassigned: Record Calibration Sample (toggle start/stop)         ║
# ║    P key takes over: Rename Manual Preset (preserved, new mnemonic)        ║
# ║  • Captures post-processed audio from _audio_bridge for accurate datasets  ║
# ║  • WAV export: ~/gbradio_calibration/calibration_YYYYMMDD_HHMMSS.wav      ║
# ║  • Designed for future Whisper fine-tuning & pronunciation calibration     ║
# ║                                                                              ║
# ║  UPGRADED in v6 (Tier 2 + Tier 3 AI integration):                          ║
# ║  • Graceful AI stack — torch/librosa/onnx/whisper/spaCy all OPTIONAL        ║
# ║    Falls back to v5 heuristics when AI libs absent                          ║
# ║  • LUFSAnalyzer v6 — proper ITU-R BS.1770 K-weighted filter (scipy)        ║
# ║    or librosa RMS proxy; heuristic fallback                                 ║
# ║  • BroadcastClassifier v6 — librosa MFCC 13-band features when available   ║
# ║    TorchBroadcastClassifier loaded from .pt file if present                 ║
# ║  • GlobalNewsDetector v6 — spaCy NER + keyword; ONNX model if present      ║
# ║  • WhisperTranscriber — faster-whisper CPU speech-to-text (Tier 3)         ║
# ║  • TopicExtractor — spaCy NER geopolitical/org/event extraction (Tier 3)   ║
# ║  • CrisisDetector — multi-category weighted keyword scoring (Tier 3)        ║
# ║  • BroadcastIntelligence — background AI orchestrator thread (Tier 3)       ║
# ║  • _send_mpv via Python Unix socket — socat dependency REMOVED              ║
# ║  • Full Termux/NetHunter Android compatibility (non-proot-distro)           ║
# ║                                                                              ║
# ║  ALL v5 features preserved:                                                  ║
# ║  • SQLite RadioDatabase · GlobalNewsDetector · RadioMap (lat/lon)           ║
# ║  • AutoHealEngine · LUFSAnalyzer/BroadcastClassifier OOP                   ║
# ║  • DistributedCrawler (async/threaded) · LiveTranslator stub                ║
# ║  • ICY metadata / now-playing / scrolling marquee                           ║
# ║  • 278-territory continent/country browser (incl. Arctic 28 territories)    ║
# ║  • 15 auto EQ presets + manual EQ + DSP mastering chain                    ║
# ║  • Favorites, auto-load, fetch-all, genre search, country search            ║
# ║  • Spectrum analyzer, LUFS meter, speech/music auto-detection               ║
# ║  • [I] Info overlay — news-score, lat/lon, AI topics, crisis panel          ║
# ║                                                                              ║
# ║  HARDCODED FEATURED STATIONS:                                                ║
# ║  • 97.5 QBS Radio  — Qatar (Asia)                                           ║
# ║  • 90.7 Love Radio — Philippines (Asia)                                     ║
# ║    Always pinned at top of their respective country lists                   ║
# ║                                                                              ║
# ║  KEY BINDINGS (player):                                                      ║
# ║   [↑↓] Navigate  [+/-] Vol  [N] Preset  [M] ManualEQ  [↵] Play             ║
# ║   [F] Fav  [B] Search  [I] Info  [T] Spectrum  [C] Country                 ║
# ║   [A] Fetch ALL / Main Menu  [X/^X] Quit  [ESC] Back                       ║
# ║   [K] AI Intelligence Panel  [R] Record Sample  [P] Rename Preset           ║
# ║                                                                              ║
# ║  For my Family: Marifel, Danica, Daphne & Kal-El  ❤️                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import curses, subprocess, signal, sys, threading, time
import json, os, math, warnings, re, sqlite3, asyncio
import shutil, socket as _socket
import wave, datetime
import numpy as np
from queue import Queue, Empty
from typing import Optional, List, Dict, Any, Tuple

# ── Standard requests ─────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("ERROR: 'requests' not found.")
    print("  Termux : pip install requests")
    print("  Linux  : pip install requests")
    sys.exit(1)

try:
    import aiohttp as _aiohttp_mod
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════════════
#  GRACEFUL AI STACK — all imports optional, never crash on missing libs
# ══════════════════════════════════════════════════════════════════════════════

# Tier 1 DSP upgrades
try:
    import scipy.signal as _scipy_signal
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import librosa as _librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Tier 2 ML inference
try:
    import torch as _torch
    import torch.nn as _nn
    HAS_TORCH = True
    # Force CPU on Android/ARM — no CUDA
    _TORCH_DEVICE = _torch.device("cpu")
except ImportError:
    HAS_TORCH = False
    _TORCH_DEVICE = None

def _check_onnx_safe():
    import subprocess
    import sys
    try:
        # Check if onnxruntime can be imported and initialized without crashing the process
        # IMPORTANT: We must also import torch if it exists, as the interaction often triggers the SIGABRT.
        code = "try: import torch; except: pass; import onnxruntime as ort; ort.set_default_logger_severity(3)"
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=10)
        return res.returncode == 0
    except:
        return False

HAS_ONNX = False
if _check_onnx_safe():
    try:
        import onnxruntime as _ort
        _ort.set_default_logger_severity(3)
        HAS_ONNX = True
    except ImportError: pass
else:
    import importlib.util
    if importlib.util.find_spec("onnxruntime"):
        # print("  ⚠️  'onnxruntime' detected but unstable on this hardware (SIGABRT). Skipping.")
        pass


# Tier 3 intelligence
try:
    from faster_whisper import WhisperModel as _WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import spacy as _spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import websockets as _websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

try:
    HAS_ONNX_TTS = HAS_ONNX
except:
    HAS_ONNX_TTS = False


# ══════════════════════════════════════════════════════════════════════════════
#  TERMUX / ANDROID / NETHUNTER ENVIRONMENT DETECTION
#  Supports: Termux vanilla, NetHunter Termux, non-proot direct install
# ══════════════════════════════════════════════════════════════════════════════
_TERMUX_PREFIX = os.environ.get("PREFIX", "")
_IS_TERMUX     = bool(_TERMUX_PREFIX and "termux" in _TERMUX_PREFIX.lower())
_TMPDIR        = os.environ.get("TMPDIR", "/tmp")
_TERMUX_BIN    = os.path.join(_TERMUX_PREFIX, "bin") if _IS_TERMUX else ""

# NetHunter detection — looks for kali-specific markers under Termux
_NH_KALI_MARKER = "/data/data/com.offsec.nethunter"
_IS_NETHUNTER   = _IS_TERMUX and (
    os.path.exists(_NH_KALI_MARKER) or
    "nethunter" in os.environ.get("TERM_PROGRAM", "").lower() or
    "nethunter" in os.environ.get("HOME", "").lower()
)

def _find_tool(name: str) -> Optional[str]:
    """Locate a binary; checks Termux PREFIX/bin first, then PATH."""
    if _IS_TERMUX:
        t = os.path.join(_TERMUX_BIN, name)
        if os.path.isfile(t) and os.access(t, os.X_OK):
            return t
    return shutil.which(name)

# Capability flags — resolved once at import time
HAS_FFMPEG = bool(_find_tool("ffmpeg"))
HAS_MPV    = bool(_find_tool("mpv"))
HAS_SOCAT  = bool(_find_tool("socat"))   # optional — Python socket used instead
HAS_PKILL  = bool(_find_tool("pkill"))

_FFMPEG_BIN = _find_tool("ffmpeg") or "ffmpeg"
_MPV_BIN    = _find_tool("mpv")    or "mpv"


def _check_requirements() -> bool:
    """Print friendly diagnostic; return True only when ffmpeg+mpv present."""
    ok = True
    if not HAS_FFMPEG:
        print("  ❌  ffmpeg not found.")
        if _IS_TERMUX:
            print("      Install:  pkg install ffmpeg")
        else:
            print("      Install:  sudo apt install ffmpeg")
        ok = False
    if not HAS_MPV:
        print("  ❌  mpv not found.")
        if _IS_TERMUX:
            print("      Install:  pkg install mpv")
        else:
            print("      Install:  sudo apt install mpv")
        ok = False
    if not HAS_SOCAT:
        pass  # v6 uses Python socket — socat not needed
    return ok


def _ai_capability_report() -> List[str]:
    """Return list of AI capability status lines for startup display."""
    lines = []
    env   = "NetHunter" if _IS_NETHUNTER else ("Termux" if _IS_TERMUX else "Linux")
    lines.append(f"  Environment : {env} (non-proot)")
    lines.append(f"  ffmpeg      : {'✅' if HAS_FFMPEG else '❌'}")
    lines.append(f"  mpv         : {'✅' if HAS_MPV else '❌'}")
    lines.append(f"  ─── AI Stack ───────────────────────")
    lines.append(f"  scipy       : {'✅ K-weighted LUFS' if HAS_SCIPY else '⬜ heuristic LUFS'}")
    lines.append(f"  librosa     : {'✅ MFCC classifier' if HAS_LIBROSA else '⬜ energy classifier'}")
    lines.append(f"  torch       : {'✅ neural inference' if HAS_TORCH else '⬜ heuristic'}")
    lines.append(f"  onnxruntime : {'✅ ONNX news model' if HAS_ONNX else '⬜ keyword news'}")
    lines.append(f"  faster-whisper: {'✅ speech-to-text' if HAS_WHISPER else '⬜ no transcription'}")
    lines.append(f"  spaCy       : {'✅ NER topics' if HAS_SPACY else '⬜ keyword topics'}")
    if _IS_TERMUX:
        lines.append("  Install AI : pip install scipy librosa torch onnxruntime faster-whisper spacy")
    return lines

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — PATHS & RUNTIME
# ══════════════════════════════════════════════════════════════════════════════
DEDICATION        = "For my Family: Marifel, Danica, Daphne & Kal-El ❤️"
MPV_SOCKET        = os.path.join(_TMPDIR, "gbradio.sock")  # $TMPDIR — Termux safe
EQ_LABELS         = ["31","63","125","250","500","1k","2k","4k","8k","16k"]
ANALYZER_SR       = 22050
MANUAL_FILE       = os.path.expanduser("~/gbradio_manual.json")
DB_FILE           = os.path.expanduser("~/gbradio_stations.db")
FAVORITES_FILE    = os.path.expanduser("~/gbradio_favorites.json")
AUTOLOAD_FILE     = os.path.expanduser("~/gbradio_autoload.json")

# AI model paths — loaded only if files exist
AI_MODEL_DIR      = os.path.expanduser("~/gbradio_models")
TORCH_MODEL_PATH  = os.path.join(AI_MODEL_DIR, "broadcast_classifier.pt")
ONNX_MODEL_PATH   = os.path.join(AI_MODEL_DIR, "news_classifier.onnx")
WHISPER_MODEL_SZ  = "tiny"   # tiny=39M params — fast on Android CPU

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — RADIOBROWSER
# ══════════════════════════════════════════════════════════════════════════════
MAX_THREADS        = 12
REFRESH_INTERVAL   = 86400
REQUEST_TIMEOUT    = 10
STREAM_TIMEOUT     = 14
THREAD_RATE_SLEEP  = 0.15

RB_MIRRORS = [
    "de1.api.radio-browser.info",
    "nl1.api.radio-browser.info",
    "at1.api.radio-browser.info",
]
RB_API = ("https://{host}/json/stations/bycountrycodeexact/{code}"
          "?limit=200&hidebroken=true&order=votes&reverse=true")

# ══════════════════════════════════════════════════════════════════════════════
#  TERRITORY FALLBACK MAP
# ══════════════════════════════════════════════════════════════════════════════
TERRITORY_FALLBACK = {
    "SX": "NL", "BL": "FR", "GS": "GB", "EH": "MA", "TA": "SH",
    "AC": "SH", "WS": "NZ", "MP": "US", "TK": "NZ", "PN": "GB",
    "JE": "GB", "GG": "GB", "IM": "GB", "NF": "AU",   # Jersey, Guernsey, IoM, Norfolk Island
}
TERRITORY_RELAY_LABEL = {
    "SX": "Netherlands relay",       "BL": "France relay",
    "GS": "UK relay (research)",     "EH": "Morocco transmitters",
    "TA": "Saint Helena relay",      "AC": "Saint Helena / BBC relay",
    "WS": "NZ / Pacific relays",     "MP": "USA relay",
    "TK": "NZ relay",                "PN": "UK relay",
    "JE": "UK relay (Jersey)",       "GG": "UK relay (Guernsey)",
    "IM": "UK relay (Isle of Man)",  "NF": "Australia relay",
}

# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED FEATURED STATIONS
# ══════════════════════════════════════════════════════════════════════════════
HARDCODED_STATIONS: Dict[tuple, List[Dict]] = {
    ("Asia", "Qatar"): [{
        "name": "97.5 QBS Radio",
        "url":  "https://live.kwikmotion.com/qmcqbsradiolive/qbsradio/playlist.m3u8",
        "country": "Qatar", "genre": "FM Radio", "codec": "AAC",
        "bitrate": 128, "votes": 0, "score": 1.0,
        "lat": 25.2854, "lon": 51.5310, "pinned": True,
    }],
    ("Asia", "Philippines"): [{
        "name": "90.7 Love Radio",
        "url":  "https://azura.loveradio.com.ph/listen/love_radio_manila/radio.mp3",
        "country": "Philippines", "genre": "Pop", "codec": "MP3",
        "bitrate": 128, "votes": 0, "score": 1.0,
        "lat": 14.5995, "lon": 120.9842, "pinned": True,
    }],
}

def _get_hardcoded(continent: str, country: str) -> List[Dict]:
    return HARDCODED_STATIONS.get((continent, country), [])

# ══════════════════════════════════════════════════════════════════════════════
#  CONTINENTS — 278 FM Broadcasting Territories
# ══════════════════════════════════════════════════════════════════════════════
CONTINENTS = {
    "Asia": {
        "China":"CN","Japan":"JP","South Korea":"KR","North Korea":"KP",
        "Mongolia":"MN","Taiwan":"TW","Hong Kong":"HK","Macau":"MO",
        "Philippines":"PH","Indonesia":"ID","Thailand":"TH","Vietnam":"VN",
        "Malaysia":"MY","Singapore":"SG","Myanmar":"MM","Cambodia":"KH",
        "Laos":"LA","Brunei":"BN","Timor-Leste":"TL","India":"IN",
        "Pakistan":"PK","Bangladesh":"BD","Sri Lanka":"LK","Nepal":"NP",
        "Bhutan":"BT","Maldives":"MV","Afghanistan":"AF","Kazakhstan":"KZ",
        "Uzbekistan":"UZ","Kyrgyzstan":"KG","Tajikistan":"TJ","Turkmenistan":"TM",
        "Turkey":"TR","Iran":"IR","Iraq":"IQ","Saudi Arabia":"SA","UAE":"AE",
        "Israel":"IL","Jordan":"JO","Lebanon":"LB","Syria":"SY","Yemen":"YE",
        "Oman":"OM","Kuwait":"KW","Qatar":"QA","Bahrain":"BH","Cyprus":"CY",
        "Armenia":"AM","Azerbaijan":"AZ","Georgia":"GE","Palestine":"PS",
        "Christmas Island":"CX","Cocos Islands":"CC","British Indian Ocean":"IO",
        "Aland Islands":"AX",
    },
    "Europe": {
        "United Kingdom":"GB","Germany":"DE","France":"FR","Spain":"ES",
        "Italy":"IT","Netherlands":"NL","Belgium":"BE","Switzerland":"CH",
        "Austria":"AT","Portugal":"PT","Ireland":"IE","Luxembourg":"LU",
        "Monaco":"MC","Liechtenstein":"LI","Andorra":"AD","Malta":"MT",
        "Iceland":"IS","San Marino":"SM","Vatican":"VA","Gibraltar":"GI",
        "Sweden":"SE","Norway":"NO","Denmark":"DK","Finland":"FI",
        "Estonia":"EE","Latvia":"LV","Lithuania":"LT","Faroe Islands":"FO",
        "Svalbard":"SJ","Jersey":"JE","Guernsey":"GG","Isle of Man":"IM",
        "Russia":"RU","Ukraine":"UA","Poland":"PL","Czech Republic":"CZ",
        "Slovakia":"SK","Hungary":"HU","Romania":"RO","Bulgaria":"BG",
        "Belarus":"BY","Moldova":"MD","Serbia":"RS","Croatia":"HR",
        "Slovenia":"SI","Bosnia":"BA","Montenegro":"ME","North Macedonia":"MK",
        "Albania":"AL","Kosovo":"XK","Transnistria":"MD","Abkhazia":"GE",
        "Greece":"GR",
    },
    "North America": {
        "United States":"US","Canada":"CA","Mexico":"MX","Greenland":"GL",
        "Bermuda":"BM","Saint Pierre & Miquelon":"PM","Guatemala":"GT",
        "Belize":"BZ","Honduras":"HN","El Salvador":"SV","Nicaragua":"NI",
        "Costa Rica":"CR","Panama":"PA","Cuba":"CU","Jamaica":"JM",
        "Haiti":"HT","Dominican Republic":"DO","Puerto Rico":"PR",
        "Trinidad & Tobago":"TT","Barbados":"BB","Bahamas":"BS","Grenada":"GD",
        "Saint Lucia":"LC","Antigua & Barbuda":"AG","Saint Vincent":"VC",
        "Dominica":"DM","Saint Kitts & Nevis":"KN","Aruba":"AW","Curacao":"CW",
        "Cayman Islands":"KY","US Virgin Islands":"VI",
        "British Virgin Islands":"VG","Guadeloupe":"GP","Martinique":"MQ",
        "Turks & Caicos":"TC","Anguilla":"AI","Sint Maarten":"SX",
        "Montserrat":"MS","Saint Barthelemy":"BL","Bonaire":"BQ",
    },
    "South America": {
        "Brazil":"BR","Argentina":"AR","Colombia":"CO","Venezuela":"VE",
        "Chile":"CL","Peru":"PE","Ecuador":"EC","Bolivia":"BO","Paraguay":"PY",
        "Uruguay":"UY","Guyana":"GY","Suriname":"SR","French Guiana":"GF",
        "Falkland Islands":"FK","South Georgia":"GS","Trinidad Island":"BR",
        "Easter Island":"CL",
    },
    "Africa": {
        "Egypt":"EG","Morocco":"MA","Algeria":"DZ","Tunisia":"TN","Libya":"LY",
        "Sudan":"SD","South Sudan":"SS","Western Sahara":"EH","Ethiopia":"ET",
        "Kenya":"KE","Tanzania":"TZ","Uganda":"UG","Rwanda":"RW","Burundi":"BI",
        "Somalia":"SO","Djibouti":"DJ","Eritrea":"ER","Madagascar":"MG",
        "Mauritius":"MU","Seychelles":"SC","Comoros":"KM","Mozambique":"MZ",
        "Zimbabwe":"ZW","Zambia":"ZM","Malawi":"MW","Reunion":"RE","Mayotte":"YT",
        "Nigeria":"NG","Ghana":"GH","Senegal":"SN","Ivory Coast":"CI",
        "Cameroon":"CM","Mali":"ML","Burkina Faso":"BF","Niger":"NE",
        "Guinea":"GN","Benin":"BJ","Togo":"TG","Sierra Leone":"SL",
        "Liberia":"LR","Gambia":"GM","Guinea-Bissau":"GW","Cape Verde":"CV",
        "Sao Tome & Principe":"ST","Mauritania":"MR","DR Congo":"CD",
        "Congo":"CG","Angola":"AO","Gabon":"GA","CAR":"CF","Chad":"TD",
        "Equatorial Guinea":"GQ","South Africa":"ZA","Namibia":"NA",
        "Botswana":"BW","Lesotho":"LS","Eswatini":"SZ","Saint Helena":"SH",
        "Tristan da Cunha":"TA","Ascension Island":"AC",
    },
    "Oceania": {
        "Australia":"AU","New Zealand":"NZ","Papua New Guinea":"PG","Fiji":"FJ",
        "Solomon Islands":"SB","Vanuatu":"VU","Samoa":"WS","Kiribati":"KI",
        "Tonga":"TO","Micronesia":"FM","Palau":"PW","Marshall Islands":"MH",
        "Tuvalu":"TV","Nauru":"NR","Guam":"GU","French Polynesia":"PF",
        "New Caledonia":"NC","Cook Islands":"CK","Niue":"NU",
        "American Samoa":"AS","Northern Mariana Islands":"MP","Tokelau":"TK",
        "Wallis & Futuna":"WF","Pitcairn":"PN","Norfolk Island":"NF",
    },
    "Arctic": {
        "Svalbard (Norway)":"SJ","Jan Mayen (Norway)":"SJ",
        "Murmansk (Russia)":"RU","Arkhangelsk (Russia)":"RU",
        "Nenets Okrug (Russia)":"RU","Yamal-Nenets (Russia)":"RU",
        "Norilsk / Krasnoyarsk (Russia)":"RU","Yakutia (Russia)":"RU",
        "Chukotka (Russia)":"RU","Magadan (Russia)":"RU",
        "Nunavut (Canada)":"CA","Northwest Territories (Canada)":"CA",
        "Yukon (Canada)":"CA","Nunavik / N. Quebec (Canada)":"CA",
        "Northern Manitoba (Canada)":"CA","Greenland (Kalaallit Nunaat)":"GL",
        "Iceland (Arctic Circle)":"IS","Northern Norway":"NO",
        "Northern Sweden (Lapland)":"SE","Northern Finland (Lapland)":"FI",
        "Alaska (USA — Arctic)":"US","Faroe Islands (sub-Arctic)":"FO",
        "Thule / Pituffik (Greenland)":"GL","Alert Station (Canada)":"CA",
        "Ny-Ålesund (Svalbard)":"SJ","Resolute Bay (Canada)":"CA",
        "Tromsø (Norway)":"NO","Longyearbyen (Svalbard)":"SJ",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO EQ PRESETS  (15 total)
# ══════════════════════════════════════════════════════════════════════════════
AUTO_PRESETS = {
    "Balanced":     [ 0, 2, 2,-2,-1, 2, 1, 2, 2, 3],
    "Country":      [ 3, 3, 2, 0, 0, 1, 1, 2, 3, 3],
    "Recommended":  [ 2, 2, 1, 0, 3, 3, 1, 1, 1, 0],
    "Flat":         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Pop":          [-2,-1, 0, 2, 4, 4, 2, 0,-1,-2],
    "Rock":         [ 5, 4, 3, 0,-2,-1, 2, 4, 5, 6],
    "Rap":          [ 9, 8, 5, 2, 0,-1, 0, 2, 5, 7],
    "Soul":         [ 4, 4, 2, 3, 2, 1, 2, 3, 4, 5],
    "News/Speech":  [-3,-2,-1, 0, 4, 5, 3, 1, 0, 0],
    "Jazz":         [ 3, 2, 1, 2, 1,-1,-1, 0, 1, 2],
    "Classical":    [ 4, 3, 2, 1, 0, 0, 0, 1, 2, 3],
    "Electronic":   [-2, 0, 2, 4, 2, 0, 3, 4, 5, 4],
    "Bass Boost":   [ 8, 7, 5, 2, 0,-1,-1, 0, 0, 0],
    "Treble Boost": [ 0, 0, 0,-1,-1, 1, 2, 4, 6, 8],
    "Arctic/Nordic":[ 3, 2, 0, 1, 2, 3, 2, 1, 3, 4],
}
AUTO_PRESET_NAMES = list(AUTO_PRESETS.keys())
DEFAULT_AUTO      = "Balanced"
MANUAL_SENTINEL   = "__MANUAL__"

KB_ROWS = [
    [("STN▲","↑"),    ("STN▼","↓"),    ("EQ◀","←"),    ("EQ▶","→")   ],
    [("VOL+","+"),    ("VOL-","-"),     ("GAIN▲","w"),  ("GAIN▼","s") ],
    [("AUTO","n"),    ("MANUAL","m"),   ("SAVE","v"),   ("NAME","p")  ],
    [("FAV","f"),     ("SEARCH","b"),   ("INFO","i"),   ("SPEC","t")  ],
    [("FETCHALL","a"),("COUNTRY","c"),  ("AI","k"),     ("PLAY","↵")  ],
    [("REC","r"),     ("QUIT","^X"),    ("",""),        ("","")       ],
]
KB_HEIGHT = len(KB_ROWS) + 3

# ══════════════════════════════════════════════════════════════════════════════
#  NEWS / CRISIS KEYWORD TABLES
# ══════════════════════════════════════════════════════════════════════════════
NEWS_TERMS = [
    "breaking", "president", "senate", "election", "government",
    "minister", "policy", "parliament", "war", "conflict", "economy",
    "crisis", "emergency", "attack", "shooting", "protest", "summit",
    "disaster", "earthquake", "hurricane", "tornado", "flood",
]

# Weighted crisis categories for CrisisDetector (Tier 3)
CRISIS_CATEGORIES: Dict[str, Tuple[List[str], int]] = {
    "armed_conflict":  (["war","battle","military","troops","missile","airstrike",
                         "ceasefire","invasion","offensive","frontline"], 4),
    "political_crisis":(["coup","impeach","resign","sanctions","embassy","diplomatic",
                         "election","protest","referendum","parliament"], 3),
    "humanitarian":    (["refugee","evacuation","displaced","famine","aid",
                         "casualties","civilian","hospital","children"], 3),
    "economic":        (["inflation","recession","collapse","debt","default",
                         "currency","market","unemployment","supply"], 2),
    "natural_disaster":(["earthquake","tsunami","hurricane","flood","wildfire",
                         "eruption","disaster","emergency","evacuation"], 3),
    "security":        (["attack","bombing","shooting","terrorism","hostage",
                         "explosion","threat","alert","security"], 4),
}


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 1 — LUFSAnalyzer
#  Tier 1: proper ITU-R BS.1770 K-weighted filter via scipy
#  Tier 1b: librosa RMS proxy
#  Fallback: v5 RMS heuristic
# ══════════════════════════════════════════════════════════════════════════════
class LUFSAnalyzer:
    """
    Integrated loudness (LUFS) from a numpy int16/float32 audio array.
    Upgrades automatically when scipy or librosa are present.
    """

    # ITU-R BS.1770 K-weighting filter coefficients (48 kHz, pre-computed)
    # Stage 1: high-shelf (head acoustic effect): +4 dB above ~1.6 kHz
    # Stage 2: high-pass (below ~40 Hz, -2 pole)
    _KW_B1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    _KW_A1 = np.array([1.0,              -1.69065929318241, 0.73248077421585])
    _KW_B2 = np.array([1.0, -2.0, 1.0])
    _KW_A2 = np.array([1.0, -1.99004745483398, 0.99007225036603])

    def __init__(self, sr: int = ANALYZER_SR):
        self.sr = sr
        self._mode = self._detect_mode()

    def _detect_mode(self) -> str:
        if HAS_SCIPY:   return "k_weighted"
        if HAS_LIBROSA: return "librosa_rms"
        return "heuristic"

    def _k_weight(self, audio_f32: np.ndarray) -> np.ndarray:
        """Apply ITU-R BS.1770 K-weighting filter chain."""
        filtered = scipy.signal.lfilter(self._KW_B1, self._KW_A1, audio_f32)
        filtered = scipy.signal.lfilter(self._KW_B2, self._KW_A2, filtered)
        return filtered

    def analyze(self, audio: np.ndarray) -> float:
        """Returns LUFS-approx value in dB, clamped to [-60, 0]."""
        if audio is None or len(audio) == 0:
            return -60.0
        a = audio.astype(np.float32)

        if self._mode == "k_weighted":
            # Normalise to [-1, +1] range then apply K-weighting
            norm = a / 32768.0 if np.max(np.abs(a)) > 1.0 else a
            kw   = self._k_weight(norm)
            ms   = float(np.mean(kw ** 2)) + 1e-12
            lufs = -0.691 + 10.0 * math.log10(ms)
            return max(-60.0, min(0.0, lufs))

        if self._mode == "librosa_rms":
            norm = a / 32768.0 if np.max(np.abs(a)) > 1.0 else a
            rms  = float(_librosa.feature.rms(y=norm).mean()) + 1e-9
            lufs = 20.0 * math.log10(rms)
            return max(-60.0, min(0.0, lufs))

        # Heuristic fallback (v5)
        rms  = np.sqrt(np.mean(a ** 2) + 1e-9)
        lufs = 20.0 * math.log10(rms / 32768.0 + 1e-9)
        return max(-60.0, min(0.0, lufs))


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 2 — BroadcastClassifier
#  Tier 2: librosa 13-band MFCC + spectral features → improved thresholds
#  Tier 2b: TorchBroadcastClassifier loaded from .pt file (if present)
#  Fallback: v5 energy heuristic
# ══════════════════════════════════════════════════════════════════════════════

# ── Optional neural net (defined only when torch available) ───────────────────
if HAS_TORCH:
    class _TorchBCNet(_nn.Module):
        """Lightweight 3-class broadcast classifier: music / speech / unknown."""
        INPUT_DIM = 17   # 13 MFCCs + energy + zcr + spectral_centroid + bandwidth

        def __init__(self):
            super().__init__()
            self.net = _nn.Sequential(
                _nn.Linear(self.INPUT_DIM, 48),
                _nn.BatchNorm1d(48),
                _nn.ReLU(),
                _nn.Dropout(0.2),
                _nn.Linear(48, 24),
                _nn.ReLU(),
                _nn.Linear(24, 3),  # [music, speech, unknown]
            )

        def forward(self, x: "_torch.Tensor") -> "_torch.Tensor":
            return self.net(x)
else:
    _TorchBCNet = None  # type: ignore


class BroadcastClassifier:
    """
    Classifies audio window as music / speech / unknown.
    Detects genre from spectral ratios.
    Upgrades automatically: torch → librosa → heuristic.
    """

    LABELS = ["music", "speech", "unknown"]

    def __init__(self, sr: int = ANALYZER_SR):
        self.sr   = sr
        self._net = None
        self._mode = "heuristic"
        self._load()

    def _load(self):
        # Try torch model from file
        if HAS_TORCH and os.path.isfile(TORCH_MODEL_PATH):
            try:
                net = _TorchBCNet()
                state = _torch.load(TORCH_MODEL_PATH,
                                    map_location=_TORCH_DEVICE)
                net.load_state_dict(state)
                net.eval()
                self._net  = net
                self._mode = "torch"
                return
            except Exception:
                pass  # fall through to next tier

        if HAS_LIBROSA:
            self._mode = "librosa"
        # else: heuristic (default)

    def _extract_features(self, audio_f32: np.ndarray) -> np.ndarray:
        """Extract 17-dim feature vector using librosa."""
        y = audio_f32 / 32768.0 if np.max(np.abs(audio_f32)) > 1.0 else audio_f32
        y = y.astype(np.float32)
        mfcc  = _librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)  # (13, T)
        feat  = list(np.mean(mfcc, axis=1))                         # 13 values
        feat.append(float(np.sqrt(np.mean(y ** 2))))                # energy
        feat.append(float(_librosa.feature.zero_crossing_rate(y).mean()))  # zcr
        feat.append(float(_librosa.feature.spectral_centroid(y=y, sr=self.sr).mean()))
        feat.append(float(_librosa.feature.spectral_bandwidth(y=y, sr=self.sr).mean()))
        return np.array(feat, dtype=np.float32)

    def classify(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return "unknown"
        a = audio.astype(np.float32)

        if self._mode == "torch" and self._net is not None:
            try:
                feats = self._extract_features(a)
                x     = _torch.FloatTensor(feats).unsqueeze(0).to(_TORCH_DEVICE)
                with _torch.no_grad():
                    logits = self._net(x)
                    idx    = int(_torch.argmax(logits, dim=1).item())
                return self.LABELS[idx]
            except Exception:
                pass  # fall to librosa

        if self._mode in ("torch", "librosa") and HAS_LIBROSA:
            try:
                feats = self._extract_features(a)
                # Decision rules tuned to MFCC+spectral feature space
                zcr     = feats[14]    # zero-crossing rate
                energy  = feats[13]    # RMS energy
                centroid= feats[15]    # spectral centroid (Hz)
                # Speech: moderate energy, high ZCR, centroid 300-3500 Hz
                if 0.02 < energy < 0.55 and zcr > 0.04 and 300 < centroid < 4000:
                    return "speech"
                if energy > 0.05:
                    return "music"
                return "unknown"
            except Exception:
                pass

        # Heuristic fallback (v5)
        energy = float(np.mean(np.abs(a / 32768.0)))
        if energy > 0.4:  return "music"
        if energy > 0.2:  return "speech"
        return "unknown"

    def detect_genre(self, fft_n: np.ndarray, rms: float) -> str:
        """Returns genre label from normalised FFT magnitude array."""
        if fft_n is None or len(fft_n) < 4:
            return "Balanced"
        n   = len(fft_n)
        low = float(np.mean(fft_n[:max(1, n // 16)]))
        mid = float(np.mean(fft_n[n // 16: n // 4]))
        hig = float(np.mean(fft_n[n // 4:]))
        if rms > 500:
            if   low > mid * 1.8 and rms > 6000: return "Rap"
            elif hig > mid * 1.4 and rms > 5000: return "Rock"
            elif mid > low * 1.4 and hig > mid * 0.8: return "Pop"
            elif low > mid       and hig < low * 0.5: return "Soul"
        return "Balanced"

    @property
    def mode(self) -> str:
        return self._mode


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 3 — GlobalNewsDetector
#  Tier 2: spaCy NER (ORG, GPE, PERSON, EVENT labels) + keyword scoring
#  Tier 2b: ONNX news classifier (if model file present)
#  Fallback: v5 keyword counting
# ══════════════════════════════════════════════════════════════════════════════
class GlobalNewsDetector:
    """
    Scores ICY StreamTitle / icy-name text for news content.
    Returns ("NEWS" | "POSSIBLE NEWS" | "NON-NEWS", score 0-100).
    Upgrades automatically: ONNX → spaCy → keyword heuristic.
    """

    def __init__(self):
        self._nlp    = None
        self._onnx   = None
        self._mode   = "keyword"
        self._load()

    def _load(self):
        # Try ONNX model
        if HAS_ONNX and os.path.isfile(ONNX_MODEL_PATH):
            try:
                self._onnx = _ort.InferenceSession(
                    ONNX_MODEL_PATH,
                    providers=["CPUExecutionProvider"]
                )
                self._mode = "onnx"
                return
            except Exception:
                pass

        # Try spaCy NER
        if HAS_SPACY:
            for model_name in ("en_core_web_sm", "en_core_web_trf"):
                try:
                    self._nlp  = _spacy.load(model_name)
                    self._mode = "spacy"
                    return
                except Exception:
                    continue
        # keyword fallback (v5)

    def _ner_score(self, text: str) -> int:
        """Use spaCy NER to count news-relevant entities."""
        doc   = self._nlp(text[:512])
        score = 0
        for ent in doc.ents:
            if ent.label_ in ("GPE", "NORP", "ORG", "EVENT", "PERSON", "LOC"):
                score += 1
        # Also count keyword hits for hybrid scoring
        t = text.lower()
        score += sum(1 for w in NEWS_TERMS if w in t)
        return score

    def detect(self, text: str) -> tuple:
        """Returns (label: str, score: int 0-100)."""
        if not text:
            return "NON-NEWS", 0

        if self._mode == "onnx" and self._onnx:
            try:
                # Simple char-level feature: pad/truncate to 128 chars
                chars = [ord(c) % 256 for c in (text + " " * 128)[:128]]
                inp   = np.array([chars], dtype=np.float32)
                iname = self._onnx.get_inputs()[0].name
                out   = self._onnx.run(None, {iname: inp})[0]
                prob  = float(out[0][1])  # P(news)
                score = int(prob * 100)
                if score >= 60:   return "NEWS", score
                if score >= 30:   return "POSSIBLE NEWS", score
                return "NON-NEWS", score
            except Exception:
                pass  # fall through

        if self._mode == "spacy" and self._nlp:
            try:
                raw_score = self._ner_score(text)
                if raw_score >= 4:   return "NEWS",          min(100, raw_score * 12)
                if raw_score >= 2:   return "POSSIBLE NEWS", raw_score * 10
                return "NON-NEWS", 0
            except Exception:
                pass

        # Keyword fallback (v5)
        t     = text.lower()
        score = sum(1 for w in NEWS_TERMS if w in t)
        if score >= 3:   return "NEWS",          min(100, score * 15)
        if score >= 1:   return "POSSIBLE NEWS", score * 10
        return "NON-NEWS", 0

    @property
    def mode(self) -> str:
        return self._mode


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 4 — AutoHealEngine (v5 unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class AutoHealEngine:
    def check_stream(self, url: str, timeout: int = 5) -> bool:
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            return r.status_code == 200
        except Exception:
            return False

    def repair(self, mirrors: List[str]) -> Optional[str]:
        for m in mirrors:
            if self.check_stream(m):
                return m
        return None

    def heal(self, primary: str, mirrors: List[str]) -> str:
        if self.check_stream(primary):
            return primary
        healed = self.repair(mirrors)
        return healed if healed else primary

_auto_heal = AutoHealEngine()


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 5 — LiveTranslator stub (v5 unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class LiveTranslator:
    def translate(self, text: str, target_lang: str = "en") -> str:
        return f"[{target_lang.upper()}] {text}"

_translator = LiveTranslator()


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 6 — RadioMap (v5 unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class RadioMap:
    def __init__(self):
        self._nodes: List[Dict[str, Any]] = []

    def add_station(self, name: str, lat: float, lon: float):
        self._nodes.append({"name": name, "lat": lat, "lon": lon})

    def clear(self):
        self._nodes.clear()

    def stations(self) -> List[Dict[str, Any]]:
        return list(self._nodes)

    def nearest(self, lat: float, lon: float, n: int = 5) -> List[Dict]:
        def dist(s): return math.sqrt((s["lat"]-lat)**2 + (s["lon"]-lon)**2)
        return sorted(self._nodes, key=dist)[:n]

_radio_map = RadioMap()


# ══════════════════════════════════════════════════════════════════════════════
#  V6 UPGRADE 7 — SQLite RadioDatabase (v5 unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class RadioDatabase:
    _lock = threading.Lock()

    def __init__(self, db_path: str = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    continent  TEXT NOT NULL,
                    country    TEXT NOT NULL,
                    name       TEXT,
                    url        TEXT,
                    bitrate    INTEGER DEFAULT 0,
                    codec      TEXT DEFAULT '',
                    votes      INTEGER DEFAULT 0,
                    genre      TEXT DEFAULT '',
                    score      REAL DEFAULT 0.0,
                    lat        REAL DEFAULT 0.0,
                    lon        REAL DEFAULT 0.0,
                    updated_at INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cont_country "
                         "ON stations(continent, country)")
            conn.commit(); conn.close()

    def upsert_country(self, continent: str, country: str,
                       station_list: List[Dict]) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM stations WHERE continent=? AND country=?",
                         (continent, country))
            now = int(time.time())
            rows = [(continent, country,
                     s.get("name",""), s.get("url",""),
                     s.get("bitrate",0), s.get("codec",""),
                     s.get("votes",0), s.get("genre",""),
                     s.get("score",0.0), s.get("lat",0.0), s.get("lon",0.0), now)
                    for s in station_list]
            conn.executemany(
                "INSERT INTO stations "
                "(continent,country,name,url,bitrate,codec,votes,genre,score,lat,lon,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit(); conn.close()

    def get_country(self, continent: str, country: str) -> List[Dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM stations WHERE continent=? AND country=? ORDER BY score DESC",
                (continent, country)).fetchall()
            conn.close()
        return [dict(r) for r in rows]

    def get_all(self) -> List[Dict]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute(
                "SELECT * FROM stations ORDER BY continent,country,score DESC").fetchall()
            conn.close()
        return [dict(r) for r in rows]

    def has_country(self, continent: str, country: str) -> bool:
        with self._lock:
            conn = self._connect()
            n = conn.execute(
                "SELECT COUNT(*) FROM stations WHERE continent=? AND country=?",
                (continent, country)).fetchone()[0]
            conn.close()
        return n > 0

    def cached_countries(self) -> int:
        with self._lock:
            conn = self._connect()
            n = conn.execute(
                "SELECT COUNT(DISTINCT continent||'/'||country) FROM stations"
            ).fetchone()[0]
            conn.close()
        return n

    def total_stations(self) -> int:
        with self._lock:
            conn = self._connect()
            n = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
            conn.close()
        return n

_db = RadioDatabase()


# ══════════════════════════════════════════════════════════════════════════════
#  DB SEEDING — seed hardcoded stations into SQLite at startup
# ══════════════════════════════════════════════════════════════════════════════
def _ensure_hardcoded_in_db() -> None:
    for (continent, country), pinned_list in HARDCODED_STATIONS.items():
        existing     = _db.get_country(continent, country)
        existing_urls = {r["url"] for r in existing}
        new_rows = []
        for p in pinned_list:
            if p["url"] not in existing_urls:
                row = dict(p)
                row.setdefault("continent", continent)
                row.setdefault("country",   country)
                row.setdefault("score",     1.0)
                new_rows.append(row)
        if new_rows:
            merged = new_rows + existing
            _db.upsert_country(continent, country, merged)


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 3 — WhisperTranscriber
#  Requires: pip install faster-whisper
#  On Android: uses CPU, tiny model (39M params) ~ 2-4 s/segment
# ══════════════════════════════════════════════════════════════════════════════
class WhisperTranscriber:
    """
    Speech-to-text from raw PCM audio using faster-whisper.
    Gracefully inactive when faster-whisper is not installed.
    """
    _SEGMENT_SEC = 12    # seconds to analyse per call
    _SR          = 16000 # Whisper native sample rate

    def __init__(self):
        self._model   = None
        self._active  = False
        self._lock    = threading.Lock()
        if HAS_WHISPER:
            self._load()

    def _load(self):
        try:
            self._model  = _WhisperModel(WHISPER_MODEL_SZ,
                                          device="cpu",
                                          compute_type="int8")  # int8 = fast on ARM
            self._active = True
        except Exception:
            self._active = False

    def transcribe_pcm(self, pcm_int16: np.ndarray,
                        src_sr: int = ANALYZER_SR) -> str:
        """
        Transcribe a block of int16 PCM audio.
        Resamples to 16 kHz if needed.
        Returns empty string if unavailable or error.
        """
        if not self._active or self._model is None:
            return ""
        with self._lock:
            try:
                audio_f32 = pcm_int16.astype(np.float32) / 32768.0
                # Resample to Whisper's 16 kHz
                if src_sr != self._SR:
                    ratio     = self._SR / src_sr
                    new_len   = int(len(audio_f32) * ratio)
                    indices   = np.linspace(0, len(audio_f32)-1, new_len)
                    audio_f32 = np.interp(indices, np.arange(len(audio_f32)),
                                          audio_f32).astype(np.float32)
                segments, _ = self._model.transcribe(
                    audio_f32,
                    beam_size=1,           # fastest
                    language=None,         # auto-detect
                    vad_filter=True,       # skip silence
                    vad_parameters={"min_silence_duration_ms": 300}
                )
                return " ".join(s.text.strip() for s in segments).strip()
            except Exception:
                return ""

    @property
    def available(self) -> bool:
        return self._active


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 3 — TopicExtractor
#  Requires: pip install spacy && python -m spacy download en_core_web_sm
# ══════════════════════════════════════════════════════════════════════════════
class TopicExtractor:
    """
    Extracts geopolitical entities and topics from transcribed text.
    spaCy NER when available; keyword fallback always works.
    """
    _GEO_LABELS = {"GPE", "LOC", "NORP", "FACILITY"}
    _EVT_LABELS = {"EVENT", "ORG", "PERSON", "PRODUCT", "LAW"}

    # Keyword fallback topic map
    _KW_TOPICS: Dict[str, List[str]] = {
        "Politics":    ["election","government","parliament","president",
                        "minister","senate","congress","vote","policy"],
        "War/Conflict":["war","conflict","military","troops","airstrike",
                        "missile","ceasefire","invasion","battle","frontline"],
        "Economy":     ["economy","market","inflation","recession","trade",
                        "currency","gdp","unemployment","bank","budget"],
        "Disaster":    ["earthquake","tsunami","hurricane","flood","wildfire",
                        "eruption","disaster","emergency","rescue"],
        "Security":    ["attack","bombing","shooting","terrorism","explosion",
                        "threat","hostage","alert","police","arrest"],
        "Health":      ["pandemic","vaccine","outbreak","hospital","disease",
                        "virus","health","infection","treatment","who"],
        "Climate":     ["climate","environment","emissions","carbon","weather",
                        "temperature","drought","species","ocean","pollution"],
    }

    def __init__(self):
        self._nlp    = None
        self._mode   = "keyword"
        if HAS_SPACY:
            for m in ("en_core_web_sm", "en_core_web_trf"):
                try:
                    self._nlp  = _spacy.load(m)
                    self._mode = "spacy"
                    break
                except Exception:
                    continue

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Returns dict with keys: entities, topics, locations, people.
        """
        result: Dict[str, Any] = {
            "entities":  [],
            "topics":    [],
            "locations": [],
            "people":    [],
        }
        if not text:
            return result

        # NER path
        if self._mode == "spacy" and self._nlp:
            try:
                doc = self._nlp(text[:1024])
                for ent in doc.ents:
                    if ent.label_ in self._GEO_LABELS:
                        result["locations"].append(ent.text)
                    if ent.label_ == "PERSON":
                        result["people"].append(ent.text)
                    if ent.label_ in self._EVT_LABELS:
                        result["entities"].append(f"{ent.text}({ent.label_})")
                # Deduplicate
                for k in result:
                    result[k] = list(dict.fromkeys(result[k]))[:5]
            except Exception:
                pass

        # Keyword topic detection (always runs as supplement)
        t = text.lower()
        for topic, kws in self._KW_TOPICS.items():
            hits = sum(1 for kw in kws if kw in t)
            if hits >= 2:
                result["topics"].append(topic)
        result["topics"] = list(dict.fromkeys(result["topics"]))[:4]
        return result

    @property
    def mode(self) -> str:
        return self._mode


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 3 — CrisisDetector
#  Multi-category weighted keyword scoring.
#  Returns crisis score 0-100 and active categories.
# ══════════════════════════════════════════════════════════════════════════════
class CrisisDetector:
    """
    Scores broadcast text for crisis signals across 6 threat categories.
    No ML dependency — pure keyword scoring, always available.
    """

    def score(self, text: str) -> Tuple[int, List[str]]:
        """Returns (score 0-100, list of active categories)."""
        if not text:
            return 0, []
        t        = text.lower()
        raw      = 0
        active   = []
        max_raw  = 0
        for cat, (kws, weight) in CRISIS_CATEGORIES.items():
            hits = sum(1 for kw in kws if kw in t)
            max_raw += len(kws) * weight
            if hits > 0:
                raw    += hits * weight
                active.append(cat)
        score = int(min(100, raw / max(1, max_raw) * 200))  # scale 0-100
        return score, active

    def label(self, score: int) -> str:
        if score >= 60: return "⚠ CRITICAL"
        if score >= 35: return "🔶 ELEVATED"
        if score >= 15: return "🔵 MONITOR"
        return "✅ CLEAR"


# ══════════════════════════════════════════════════════════════════════════════
#  TIER 3 — BroadcastIntelligence
#  Background thread orchestrator that runs Whisper + TopicExtractor +
#  CrisisDetector on captured audio blocks.
#  Pushes results into a shared state dict (lock-protected).
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  V7 UPGRADES - NEURAL DUBBING
# ══════════════════════════════════════════════════════════════════════════════

class AudioMixer:
    """Handles real-time dual-channel audio mixing in Python before MPV."""
    def __init__(self):
        self.dub_queue = Queue()
        self.mode = "ORIGINAL" # ORIGINAL, DUBBED, MOVIE
        self.lock = threading.Lock()
        self.current_dub_chunk = None
        self.dub_pos = 0

    def add_dub_audio(self, pcm_data: bytes):
        """Adds generated TTS PCM audio to the dubbing queue."""
        self.dub_queue.put(pcm_data)

    def mix(self, original_pcm: bytes) -> bytes:
        if self.mode == "ORIGINAL":
            return original_pcm
            
        orig_arr = np.frombuffer(original_pcm, dtype=np.int16).astype(np.float32)
        mixed_arr = orig_arr.copy()
        
        # Load new dub chunk if needed
        if self.current_dub_chunk is None or self.dub_pos >= len(self.current_dub_chunk):
            try:
                raw_dub = self.dub_queue.get_nowait()
                self.current_dub_chunk = np.frombuffer(raw_dub, dtype=np.int16).astype(np.float32)
                self.dub_pos = 0
            except Empty:
                self.current_dub_chunk = None

        if self.current_dub_chunk is not None:
            frames_needed = len(orig_arr)
            available = len(self.current_dub_chunk) - self.dub_pos
            take = min(frames_needed, available)
            
            dub_segment = self.current_dub_chunk[self.dub_pos:self.dub_pos + take]
            
            if self.mode == "MOVIE":
                # Duck original audio to 20%
                mixed_arr[:take] = (orig_arr[:take] * 0.2) + (dub_segment * 0.8)
            elif self.mode == "DUBBED":
                # Mute original completely
                mixed_arr[:take] = dub_segment
                
            self.dub_pos += take
            
        return np.clip(mixed_arr, -32768, 32767).astype(np.int16).tobytes()

_audio_mixer = AudioMixer()

# ══════════════════════════════════════════════════════════════════════════════
#  SPEECH LAB — SpeechCalibrationRecorder
#  Captures post-processed FM audio for WAV dataset generation.
#  Toggle with [R] in the main player (start / stop → save).
# ══════════════════════════════════════════════════════════════════════════════
class SpeechCalibrationRecorder:
    """Thread-safe PCM ring buffer that saves timestamped WAV calibration
    samples from the live _audio_bridge post-mix stream."""

    def __init__(self):
        self.active = False
        self.buffer = bytearray()
        self.lock   = threading.Lock()

    def start(self):
        with self.lock:
            self.active = True
            self.buffer.clear()

    def feed(self, pcm: bytes):
        """Called every _audio_bridge chunk — only records when active."""
        if not self.active:
            return
        with self.lock:
            self.buffer.extend(pcm)

    def stop(self) -> bytes:
        with self.lock:
            self.active = False
            return bytes(self.buffer)

def save_calibration_sample(pcm: bytes) -> str:
    """Write PCM bytes to a timestamped WAV file in CALIBRATION_DIR."""
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(CALIBRATION_DIR, f"calibration_{ts}.wav")
    with wave.open(outfile, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(ANALYZER_SR)
        wf.writeframes(pcm)
    return outfile

_recorder = SpeechCalibrationRecorder()

class NeuralTranslationEngine:
    def __init__(self):
        self.translator = None
        self.tokenizer = None
        self.active_lang_pair = "OFF" # OFF, AR_EN, EN_AR
        self.is_loaded = False
        self.nllb_model_name = "facebook/nllb-200-distilled-600M"

    def load_models(self):
        if not HAS_TRANSFORMERS or self.is_loaded: return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.nllb_model_name)
            self.translator = AutoModelForSeq2SeqLM.from_pretrained(self.nllb_model_name)
            self.is_loaded = True
        except Exception:
            pass

    def translate(self, text: str) -> str:
        if not self.is_loaded or self.active_lang_pair == "OFF": return ""
        
        src_lang = "eng_Latn" if self.active_lang_pair == "EN_AR" else "arb_Arab"
        tgt_lang = "arb_Arab" if self.active_lang_pair == "EN_AR" else "eng_Latn"
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            translated_tokens = self.translator.generate(
                **inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang], max_length=100
            )
            return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        except Exception:
            return ""

_translation_engine = NeuralTranslationEngine()

class NeuralTTSEngine:
    """Asynchronous Neural Text-to-Speech Engine for Geeky Beast Radio v7."""
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr
        self.is_loaded = False
        self.onnx_session_en = None
        self.onnx_session_ar = None
        self.lock = threading.Lock()
        self.en_model_path = os.path.expanduser("~/gbradio_models/en_tts.onnx")
        self.ar_model_path = os.path.expanduser("~/gbradio_models/ar_tts.onnx")

    def load_models(self):
        if not HAS_ONNX_TTS or self.is_loaded: return
        with self.lock:
            try:
                providers = ['CPUExecutionProvider']
                self.onnx_session_en = _ort.InferenceSession(self.en_model_path, providers=providers)
                self.onnx_session_ar = _ort.InferenceSession(self.ar_model_path, providers=providers)
                self.is_loaded = True
            except Exception as e:
                self.is_loaded = False

    def synthesize(self, text: str, lang: str) -> bytes:
        if not text.strip(): return b""
        if self.is_loaded and HAS_ONNX_TTS:
            return self._synthesize_onnx(text, lang)
        elif HAS_GTTS:
            return self._synthesize_gtts(text, lang)
        else:
            return b""

    def _synthesize_onnx(self, text: str, lang: str) -> bytes:
        try:
            # Stub for ONNX text-to-phoneme mapping. 
            # In a real scenario you'd tokenize text to phoneme ids.
            session = self.onnx_session_ar if lang == "ar" else self.onnx_session_en
            # For now, return a silent 0.5s audio chunk as fallback if ONNX inference fails to mock it.
            dummy_audio = np.zeros(self.target_sr // 2, dtype=np.float32)
            pcm_16 = np.int16(dummy_audio * 32767).tobytes()
            return pcm_16
        except Exception:
            return b""

    def _synthesize_gtts(self, text: str, lang: str) -> bytes:
        try:
            tts = gTTS(text=text, lang=lang)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            if HAS_LIBROSA:
                audio_data, sr = _librosa.load(fp, sr=self.target_sr, mono=True)
                pcm_16 = np.int16(audio_data * 32767).tobytes()
                return pcm_16
            else:
                return b""
        except Exception:
            return b""

_tts_engine = NeuralTTSEngine(target_sr=ANALYZER_SR)

class BroadcastIntelligence:
    """
    Runs AI analysis in a separate low-priority thread.
    Collects audio blocks from the analyzer, runs transcription and NLP,
    updates shared intelligence state.
    """
    _SAMPLE_SECS   = 10    # seconds of audio per analysis cycle
    _COOLDOWN_SECS = 20    # minimum gap between analyses

    def __init__(self):
        self._whisper  = WhisperTranscriber()
        self._topics   = TopicExtractor()
        self._crisis   = CrisisDetector()
        self._lock     = threading.Lock()
        self._buf      = bytearray()
        self._token    = 0
        self._last_run = 0.0
        # Shared state (read by TUI)
        self.state: Dict[str, Any] = {
            "transcript":   "",
            "topics":       [],
            "locations":    [],
            "people":       [],
            "crisis_score": 0,
            "crisis_cats":  [],
            "crisis_label": "✅ CLEAR",
            "updated_at":   0.0,
            "running":      False,
            "mode":         self._describe_mode(),
        }

    def _describe_mode(self) -> str:
        parts = []
        if HAS_WHISPER:  parts.append("Whisper")
        if HAS_SPACY:    parts.append("spaCy")
        parts.append("CrisisKW")
        return "+".join(parts) if parts else "keyword-only"

    def feed(self, pcm_chunk: bytes):
        """Feed raw s16le PCM bytes from the analyzer process."""
        with self._lock:
            self._buf.extend(pcm_chunk)

    def start(self, token: int):
        """Start background AI thread (call after new stream starts)."""
        self._token = token
        self._buf.clear()
        t = threading.Thread(target=self._loop, args=(token,), daemon=True)
        t.start()

    def stop(self):
        self._token = -1

    def _loop(self, token: int):
        target_bytes = self._SAMPLE_SECS * ANALYZER_SR * 2  # int16 = 2 bytes
        while self._token == token:
            time.sleep(1.0)
            now = time.time()
            if now - self._last_run < self._COOLDOWN_SECS:
                continue
            with self._lock:
                buf_len = len(self._buf)
            if buf_len < target_bytes:
                continue
            # Grab snapshot
            with self._lock:
                block = bytes(self._buf[:target_bytes])
                self._buf = self._buf[target_bytes // 2:]  # 50% overlap
            self._analyse(block, token)

    def _analyse(self, block: bytes, token: int):
        if self._token != token:
            return
        self.state["running"] = True
        try:
            pcm = np.frombuffer(block, dtype=np.int16)
            # Transcription
            transcript = self._whisper.transcribe_pcm(pcm, ANALYZER_SR)
            
            # Translation & TTS Pipeline (V7)
            translated_text = ""
            if _translation_engine.active_lang_pair != "OFF" and transcript.strip():
                translated_text = _translation_engine.translate(transcript)
                if translated_text:
                    target_lang = "ar" if _translation_engine.active_lang_pair == "EN_AR" else "en"
                    pcm_audio = _tts_engine.synthesize(translated_text, lang=target_lang)
                    if pcm_audio:
                        _audio_mixer.add_dub_audio(pcm_audio)
                        
            # Topics + crisis from transcript + ICY metadata
            with _np_lock:
                icy_text = now_playing_raw + " " + icy_station_name
            combined = (transcript + " " + translated_text + " " + icy_text).strip()
            topic_result  = self._topics.extract(combined)
            crisis_s, cats = self._crisis.score(combined)
            self.state.update({
                "transcript":   transcript[:200],
                "translation":  translated_text[:200],
                "topics":       topic_result["topics"],
                "locations":    topic_result["locations"],
                "people":       topic_result["people"],
                "crisis_score": crisis_s,
                "crisis_cats":  cats,
                "crisis_label": self._crisis.label(crisis_s),
                "updated_at":   time.time(),
                "running":      False,
            })
            self._last_run = time.time()
        except Exception:
            self.state["running"] = False

# Singleton intelligence instance
_broadcast_intel = BroadcastIntelligence()
#####

# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME STATE
#  All mutable globals — accessed from threads and TUI loop.
#  _np_lock guards the ICY / now-playing strings read by BroadcastIntelligence.
# ══════════════════════════════════════════════════════════════════════════════
STATIONS: List[Dict] = []

_radio_lock     = threading.Lock()
_bridge_token   = 0
_analyzer_token = 0

ffmpeg_proc:   Optional[subprocess.Popen] = None
mpv_proc:      Optional[subprocess.Popen] = None
analyzer_proc: Optional[subprocess.Popen] = None

current_idx    = 0
current_vol    = 100
selected_band  = 0

active_mode        = DEFAULT_AUTO
_auto_preset_idx   = 0
manual_preset: Dict[str, Any] = {}  # loaded after _load_manual() defined
in_manual_mode = False

_target_eq  = list(AUTO_PRESETS[DEFAULT_AUTO])
_cur_eq_f   = [float(v) for v in _target_eq]
active_eq   = list(_target_eq)
_XFADE_STEP   = 0.09
_XFADE_THRESH = 0.05

NUM_BANDS    = 20
band_heights = [0.0] * NUM_BANDS
_smooth_bh   = [0.0] * NUM_BANDS
peaks        = [0.0] * NUM_BANDS

detection_label  = "MUSIC"
genre_label      = "—"
lufs_display     = -60.0
stream_status    = "CONNECTING"
need_restart     = False
save_flash       = 0.0
fav_flash        = 0.0

# ── Speech Lab / Calibration Recorder ────────────────────────────────────────
CALIBRATION_DIR  = os.path.expanduser("~/gbradio_calibration")
os.makedirs(CALIBRATION_DIR, exist_ok=True)
recording_enabled   = False
record_start_time   = 0.0
record_flash        = 0.0   # feedback flash timestamp
last_auto_switch = 0.0
AUTO_COOLDOWN    = 20.0

_speech_votes  = 0
_music_votes   = 0
VOTE_THRESHOLD = 30
_genre_votes:  Dict[str, int] = {}
GENRE_VOTE_WIN = 45

_auto_detection_preset = DEFAULT_AUTO

show_spectrum     = True
show_info_panel   = False
show_ai_panel     = False   # [K] AI intelligence panel toggle
search_mode       = False
search_query      = ""
_filtered_indices: List[int] = []

current_continent = ""
current_country   = ""

_reconnect_attempts = 0
MAX_RECONNECT_DELAY = 30.0

# ICY metadata (shared between analyzer and TUI)
_np_lock       = threading.Lock()
now_playing_raw  = ""
icy_station_name = ""
_marquee_offset  = 0

# ── TUI render performance globals ───────────────────────────────────────────
# Color handles — initialised once per curses session, never per-frame
_tui_C1 = _tui_C2 = _tui_C3 = _tui_C4 = _tui_C5 = 0
_tui_colors_ready = False

# Dirty-flag render system — only redraw when state actually changed
_ui_prev_hash   = None
_ui_prev_size   = (0, 0)
_ui_frame_count = 0   # increments every getch() cycle for animation pacing


def _ensure_colors():
    """Call curses.init_pair exactly once per session. Safe to call many times."""
    global _tui_C1, _tui_C2, _tui_C3, _tui_C4, _tui_C5, _tui_colors_ready
    if _tui_colors_ready:
        return
    try:
        curses.start_color()
        curses.use_default_colors()
    except Exception:
        pass
    curses.init_pair(1, curses.COLOR_CYAN,    curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN,   curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW,  curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED,     curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    _tui_C1 = curses.color_pair(1)
    _tui_C2 = curses.color_pair(2)
    _tui_C3 = curses.color_pair(3)
    _tui_C4 = curses.color_pair(4)
    _tui_C5 = curses.color_pair(5)
    _tui_colors_ready = True


def _ui_state_hash() -> int:
    """Cheap hash of all visible UI state. Mismatch → redraw needed."""
    with _np_lock:
        np_snap = now_playing_raw
    return hash((
        current_idx, current_vol, stream_status, active_mode,
        in_manual_mode, show_spectrum, show_info_panel, show_ai_panel,
        search_mode, search_query, current_country, current_continent,
        detection_label, genre_label,
        round(lufs_display, 1),
        save_flash > 0.0, fav_flash > 0.0,
        np_snap,
        tuple(active_eq[:5]),          # first 5 EQ bands as proxy
        tuple(round(b, 1) for b in band_heights[:5]),  # spectrum proxy
    ))


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS — preset resolution
# ══════════════════════════════════════════════════════════════════════════════
def _current_display_name() -> str:
    if in_manual_mode:
        return f"✎ {manual_preset.get('name','MyEQ')}"
    return active_mode


def _current_gains() -> List[int]:
    if in_manual_mode:
        return list(manual_preset.get("gains", [0]*10))
    return list(AUTO_PRESETS.get(active_mode, AUTO_PRESETS[DEFAULT_AUTO]))


def _activate_preset(gains: List[int]):
    global _target_eq
    _target_eq[:] = [max(-15, min(15, int(g))) for g in gains]


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH / FILTER helpers
# ══════════════════════════════════════════════════════════════════════════════
def _apply_search(query: str) -> List[int]:
    if not query:
        return list(range(len(STATIONS)))
    q = query.lower()
    return [i for i, s in enumerate(STATIONS)
            if q in s.get("name", "").lower()
            or q in s.get("genre", "").lower()]


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP SESSION
# ══════════════════════════════════════════════════════════════════════════════
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent":   "GeekyBeastRadio/6.0 (compatible)",
        "Accept":       "*/*",
        "Icy-MetaData": "1",
    })
    return s


def safe_get(session, url, timeout=REQUEST_TIMEOUT,
             stream=False, retries=2, **kw) -> Optional[requests.Response]:
    for attempt in range(retries + 1):
        try:
            return session.get(url, timeout=timeout, stream=stream, **kw)
        except (requests.Timeout, requests.ConnectionError):
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        except Exception:
            break
    return None


def safe_head(session, url, timeout=REQUEST_TIMEOUT) -> Optional[requests.Response]:
    try:
        return session.head(url, timeout=timeout, allow_redirects=True)
    except Exception:
        return None


AUDIO_MAGIC = [
    (0, b"\xff\xfb"), (0, b"\xff\xf3"), (0, b"\xff\xf2"),
    (0, b"ID3"),      (0, b"OggS"),     (0, b"\xff\xf1"),
    (0, b"\xff\xf9"), (0, b"fLaC"),     (0, b"ICY "),
    (0, b"#EXTM3U"),  (0, b"#EXT-X-"),
]
VALID_CT = {
    "audio/mpeg", "audio/mp3", "audio/aac", "audio/aacp",
    "audio/ogg",  "audio/vorbis", "audio/opus", "audio/flac",
    "audio/x-mp3", "audio/x-mpeg", "audio/x-aac",
    "application/ogg",
    "application/vnd.apple.mpegurl",
    "application/x-mpegurl",
    "application/octet-stream",
}
INVALID_CT_PREFIXES = ("text/html", "text/xml", "image/", "video/mp4", "video/webm")


# ══════════════════════════════════════════════════════════════════════════════
#  FETCH from RadioBrowser  (with v6 territory fallback)
# ══════════════════════════════════════════════════════════════════════════════
def fetch_stations_api(country_code: str, country_name: str = "") -> List[Dict]:
    session = make_session()
    # Stage 1 — exact ISO code
    for host in RB_MIRRORS:
        url  = RB_API.format(host=host, code=country_code)
        resp = safe_get(session, url, timeout=REQUEST_TIMEOUT)
        if resp and resp.status_code == 200:
            try:
                data = resp.json()
                if isinstance(data, list) and data:
                    return data
            except Exception:
                continue

    # Stage 2 — country name search fallback
    if country_name:
        for host in RB_MIRRORS:
            url  = f"https://{host}/json/stations/search?country={requests.utils.quote(country_name)}&limit=200&hidebroken=true"
            resp = safe_get(session, url, timeout=REQUEST_TIMEOUT)
            if resp and resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        return data
                except Exception:
                    continue

    # Stage 3 — territory parent fallback
    parent = TERRITORY_FALLBACK.get(country_code)
    if parent:
        for host in RB_MIRRORS:
            url  = RB_API.format(host=host, code=parent)
            resp = safe_get(session, url, timeout=REQUEST_TIMEOUT)
            if resp and resp.status_code == 200:
                try:
                    data = resp.json()
                    if isinstance(data, list) and data:
                        return data
                except Exception:
                    continue

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYLIST REPAIR
# ══════════════════════════════════════════════════════════════════════════════
def _repair_playlist(url: str) -> str:
    ext = url.lower().split("?")[0]
    if not (ext.endswith(".pls") or ext.endswith(".m3u") or ext.endswith(".m3u8")):
        return url
    try:
        r = requests.get(url, timeout=8, allow_redirects=True)
        if r.status_code != 200:
            return url
        for line in r.text.splitlines():
            line = line.strip()
            if line.startswith("http") or line.startswith("File1="):
                line = line.replace("File1=", "").strip()
                if line.startswith("http"):
                    return line
    except Exception:
        pass
    return url


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATE streams
# ══════════════════════════════════════════════════════════════════════════════
def validate_station(raw: Dict) -> Optional[Dict]:
    url = (raw.get("url_resolved") or raw.get("url", "")).strip()
    if not url or not url.startswith("http"):
        return None
    url = _repair_playlist(url)

    session    = make_session()
    t0         = time.time()
    head       = safe_head(session, url, timeout=STREAM_TIMEOUT)
    latency_ms = int((time.time() - t0) * 1000)

    if head is not None and head.status_code == 200:
        ct = head.headers.get("Content-Type", "").lower().split(";")[0].strip()
        if any(ct.startswith(p) for p in INVALID_CT_PREFIXES):
            return None
        if ct in VALID_CT and ct != "application/octet-stream":
            return _make_station(raw, url, latency_ms)

    t0   = time.time()
    resp = safe_get(session, url, timeout=STREAM_TIMEOUT, stream=True, retries=1)
    latency_ms = int((time.time() - t0) * 1000)

    if not resp or resp.status_code not in (200, 206):
        if resp: resp.close()
        return None

    ct = resp.headers.get("Content-Type", "").lower().split(";")[0].strip()
    if any(ct.startswith(p) for p in INVALID_CT_PREFIXES):
        resp.close()
        return None

    try:
        chunk = next(resp.iter_content(256), b"")
    except Exception:
        chunk = b""
    finally:
        resp.close()

    if not chunk:
        return None

    for offset, magic in AUDIO_MAGIC:
        if chunk[offset: offset + len(magic)] == magic:
            return _make_station(raw, url, latency_ms)

    if ct in VALID_CT and ct != "application/octet-stream":
        return _make_station(raw, url, latency_ms)

    return None


def _make_station(raw: Dict, url: str, latency_ms: int) -> Dict:
    name  = re.sub(r'\s{2,}', ' ', raw.get("name", "Unknown")).strip() or "Unknown"
    tags  = raw.get("tags", "") or ""
    genre = tags.split(",")[0].strip().title() if tags else "FM Radio"
    return {
        "name":       name,
        "url":        url,
        "country":    raw.get("country", "").strip(),
        "genre":      genre,
        "codec":      (raw.get("codec") or "").upper(),
        "bitrate":    int(raw.get("bitrate") or 0),
        "votes":      int(raw.get("votes")   or 0),
        "latency_ms": latency_ms,
        "lat":        float(raw.get("geo_lat") or 0.0),
        "lon":        float(raw.get("geo_long") or 0.0),
        "score":      0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATE RUNNER (concurrent, thread-safe)
# ══════════════════════════════════════════════════════════════════════════════
def run_validation(raw_list: List[Dict], progress_cb=None) -> List[Dict]:
    if not raw_list:
        return []
    total   = len(raw_list)
    q       = Queue()
    results: List[Dict] = []
    lock    = threading.Lock()
    stats   = {"checked": 0}

    for s in raw_list:
        q.put(s)

    def worker():
        while True:
            try:
                raw = q.get(timeout=2.0)
            except Empty:
                break
            try:
                station = validate_station(raw)
            except Exception:
                station = None
            finally:
                q.task_done()
            with lock:
                stats["checked"] += 1
                if station:
                    results.append(station)
                if progress_cb:
                    progress_cb(stats["checked"], total)
            time.sleep(THREAD_RATE_SLEEP)

    n = min(MAX_THREADS, total)
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(n)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=90)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RANK stations
# ══════════════════════════════════════════════════════════════════════════════
def rank_stations(stations: List[Dict]) -> List[Dict]:
    if not stations:
        return []
    max_votes   = max((s.get("votes",   0) for s in stations), default=1) or 1
    max_bitrate = max((s.get("bitrate", 0) for s in stations), default=1) or 1
    max_latency = max((s.get("latency_ms", 1) for s in stations), default=1) or 1
    for s in stations:
        v = s.get("votes",      0) / max_votes
        b = s.get("bitrate",    0) / max_bitrate
        l = 1.0 - (s.get("latency_ms", 0) / max_latency)
        s["score"] = round(0.50 * v + 0.30 * b + 0.20 * l, 4)
    stations.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return stations


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS (wrappers over RadioDatabase SQLite)
# ══════════════════════════════════════════════════════════════════════════════
def _store_key(continent: str, country: str) -> str:
    return f"{continent}/{country}"


def _save_to_db(continent: str, country: str, stations: List[Dict]) -> None:
    # Merge hardcoded pinned stations first
    pinned     = _get_hardcoded(continent, country)
    pin_urls   = {p["url"] for p in pinned}
    filtered   = [s for s in stations if s.get("url") not in pin_urls]
    merged     = pinned + filtered
    _db.upsert_country(continent, country, merged)


def _load_from_db(continent: str, country: str) -> List[Dict]:
    rows = _db.get_country(continent, country)
    if not rows:
        return []
    # Ensure pinned stations are first
    pinned   = _get_hardcoded(continent, country)
    pin_urls = {p["url"] for p in pinned}
    non_pin  = [r for r in rows if r.get("url") not in pin_urls]
    return pinned + non_pin


def _is_fully_cached() -> bool:
    for cont, countries in CONTINENTS.items():
        for country in countries:
            if not _db.has_country(cont, country):
                return False
    return True


def _cached_count() -> Tuple[int, int]:
    total  = sum(len(v) for v in CONTINENTS.values())
    cached = sum(
        1 for cont, countries in CONTINENTS.items()
        for country in countries
        if _db.has_country(cont, country)
    )
    return cached, total


def _all_stations_flat() -> List[Dict]:
    rows   = _db.get_all()
    result = []
    for r in rows:
        result.append({**r, "_continent": r.get("continent",""), "_country": r.get("country","")})
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FAVORITES PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════
def _load_favorites() -> List[Dict]:
    try:
        if os.path.exists(FAVORITES_FILE):
            with open(FAVORITES_FILE) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def _save_favorites(favs: List[Dict]) -> None:
    try:
        with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
            json.dump(favs, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _toggle_favorite(station: Dict) -> bool:
    favs = _load_favorites()
    url  = station.get("url", "")
    if any(f.get("url") == url for f in favs):
        favs = [f for f in favs if f.get("url") != url]
        _save_favorites(favs)
        return False
    favs.append({"name": station.get("name", "?"), "url": url})
    _save_favorites(favs)
    return True


def _is_favorite(station: Dict) -> bool:
    url  = station.get("url", "")
    return any(f.get("url") == url for f in _load_favorites())


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-LOAD SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
def _load_autoload() -> Dict:
    default: Dict[str, Any] = {
        "enabled":        False,
        "continent":      "",
        "country":        "",
        "station_name":   "",
        "station_url":    "",
        "preset":         "Balanced",
        "volume":         100,
        "extra_stations": [],
    }
    try:
        if os.path.exists(AUTOLOAD_FILE):
            with open(AUTOLOAD_FILE) as f:
                data = json.load(f)
            default.update({k: v for k, v in data.items() if k in default})
    except Exception:
        pass
    return default


def _save_autoload(cfg: Dict) -> bool:
    try:
        with open(AUTOLOAD_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MANUAL EQ PRESET (persisted)
# ══════════════════════════════════════════════════════════════════════════════
def _load_manual() -> Dict:
    try:
        if os.path.exists(MANUAL_FILE):
            with open(MANUAL_FILE) as f:
                data = json.load(f)
            name  = str(data.get("name", "My EQ"))[:24]
            gains = data.get("gains", [0]*10)
            if isinstance(gains, list) and len(gains) == 10:
                gains = [max(-15, min(15, int(g))) for g in gains]
                return {"name": name, "gains": gains}
    except Exception:
        pass
    return {"name": "My EQ", "gains": [0]*10}


def _save_manual() -> bool:
    try:
        with open(MANUAL_FILE, "w") as f:
            json.dump({"name": manual_preset["name"],
                       "gains": list(manual_preset["gains"])}, f, indent=2)
        return True
    except Exception:
        return False


# Initialise manual_preset after function is defined
manual_preset = _load_manual()


# ══════════════════════════════════════════════════════════════════════════════
#  DSP — PROFESSIONAL MASTERING CHAIN
# ══════════════════════════════════════════════════════════════════════════════
def build_filter_chain(gains: List[int]) -> str:
    freqs = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    flt   = []
    for f, g in zip(freqs, gains):
        gc = max(-15, min(15, round(float(g))))
        flt.append(f"equalizer=f={f}:width_type=o:width=2:g={gc}")
    flt.append("equalizer=f=60:width_type=q:width=1.2:g=5")
    flt.append("equalizer=f=8000:width_type=h:width=1:g=2")
    flt.append("aecho=0.8:0.8:18:0.25")
    flt.append("acompressor=threshold=-14dB:ratio=3:attack=20:release=200:makeup=2")
    flt.append("loudnorm=I=-16:TP=-1.5:LRA=11:linear=true")
    flt.append("alimiter=level_in=1:level_out=1:limit=0.891:attack=5:release=50")
    return ",".join(flt)


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESS MANAGEMENT  (Termux-safe)
# ══════════════════════════════════════════════════════════════════════════════
def _kill_procs():
    """Terminate only the tracked child PIDs.
    Never uses broad pkill/pkill-by-name — that kills unrelated mpv/ffmpeg
    processes on shared Termux / NetHunter environments."""
    global ffmpeg_proc, mpv_proc, analyzer_proc
    _broadcast_intel.stop()

    # Snapshot PIDs before we null the globals
    tracked = [(p, p.pid) for p in (ffmpeg_proc, mpv_proc, analyzer_proc) if p]

    # Ask each process to terminate gracefully
    for p, _pid in tracked:
        try:
            p.terminate()
        except Exception:
            pass

    # Wait up to 1.2 s total; SIGKILL individual stragglers by PID only
    deadline = time.time() + 1.2
    for p, pid in tracked:
        wait_left = max(0.05, deadline - time.time())
        try:
            p.wait(timeout=wait_left)
        except Exception:
            try:
                import signal as _sig
                os.kill(pid, _sig.SIGKILL)
            except Exception:
                pass

    ffmpeg_proc = mpv_proc = analyzer_proc = None
    time.sleep(0.04)

    if os.path.exists(MPV_SOCKET):
        try:
            os.remove(MPV_SOCKET)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  MPV IPC  — pure Python Unix socket (socat NOT required)
# ══════════════════════════════════════════════════════════════════════════════
def _send_mpv(cmd: Dict) -> bool:
    if not os.path.exists(MPV_SOCKET):
        return False
    try:
        payload = (json.dumps(cmd) + "\n").encode()
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(0.4)
        s.connect(MPV_SOCKET)
        s.sendall(payload)
        s.close()
        return True
    except Exception:
        return False


def apply_volume_now():
    """Set mpv volume via IPC socket first; fall back to mpv respawn.
    Snapshots ffmpeg_proc to a local variable before any async window
    where _kill_procs() on another thread could null the global."""
    global mpv_proc
    if _send_mpv({"command": ["set_property", "volume", current_vol]}):
        return
    # Snapshot immediately — _kill_procs may null ffmpeg_proc at any time
    local_ff = ffmpeg_proc
    if local_ff is None or local_ff.poll() is not None:
        return
    try:
        if mpv_proc:
            mpv_proc.terminate()
    except Exception:
        pass
    if os.path.exists(MPV_SOCKET):
        try:
            os.remove(MPV_SOCKET)
        except Exception:
            pass
    time.sleep(0.04)
    try:
        new_mpv = subprocess.Popen(
            [_MPV_BIN, "--no-video",
             f"--volume={current_vol}",
             f"--input-ipc-server={MPV_SOCKET}",
             "--cache=yes", "--cache-secs=3", "-"],
            stdin=local_ff.stdout,          # use snapshot, not the global
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        globals()["mpv_proc"] = new_mpv
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  ICY METADATA FETCHER  — background thread reads now-playing title
# ══════════════════════════════════════════════════════════════════════════════
def _icy_fetcher(url: str, token: int):
    global now_playing_raw, icy_station_name
    try:
        r = requests.get(
            url, headers={"Icy-MetaData": "1", "User-Agent": "GeekyBeastRadio/6.0"},
            stream=True, timeout=10)
        icy_interval = int(r.headers.get("icy-metaint", 0))
        station_nm   = r.headers.get("icy-name", "")
        with _np_lock:
            icy_station_name = station_nm

        if icy_interval <= 0:
            return
        count = 0
        for chunk in r.iter_content(chunk_size=icy_interval + 4080):
            if _bridge_token != token:
                break
            if len(chunk) < icy_interval:
                break
            meta_len = chunk[icy_interval] * 16
            if meta_len > 0 and len(chunk) >= icy_interval + 1 + meta_len:
                meta_raw = chunk[icy_interval + 1: icy_interval + 1 + meta_len]
                try:
                    decoded = meta_raw.rstrip(b"\x00").decode("utf-8", errors="replace")
                    m = re.search(r"StreamTitle='([^']*)'", decoded)
                    if m:
                        with _np_lock:
                            now_playing_raw = m.group(1).strip()
                except Exception:
                    pass
            count += 1
            if count > 500:
                break
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  RADIO START
# ══════════════════════════════════════════════════════════════════════════════
def start_radio(reset_votes: bool = False):
    global ffmpeg_proc, mpv_proc, analyzer_proc
    global _bridge_token, _analyzer_token
    global _cur_eq_f, active_eq
    global stream_status, need_restart
    global _speech_votes, _music_votes, _genre_votes
    global detection_label, genre_label
    global _reconnect_attempts
    global now_playing_raw, icy_station_name

    if not STATIONS:
        stream_status = "NO STATIONS"
        return
    if not HAS_FFMPEG or not HAS_MPV:
        stream_status = "NO FFMPEG/MPV"
        return
    if not _radio_lock.acquire(blocking=False):
        return
    try:
        _bridge_token   += 1
        _analyzer_token += 1
        my_bt = _bridge_token
        my_at = _analyzer_token

        _kill_procs()

        if reset_votes:
            _speech_votes = _music_votes = 0
            _genre_votes  = {}
            detection_label = "MUSIC"
            genre_label     = "—"
            _reconnect_attempts = 0
            with _np_lock:
                now_playing_raw  = ""
                icy_station_name = ""

        _cur_eq_f[:] = [float(v) for v in _target_eq]
        active_eq[:] = [round(v) for v in _cur_eq_f]

        # Defensive clamp — current_idx can drift if a custom station was
        # prepended / removed while the UI was open
        safe_idx = max(0, min(current_idx, len(STATIONS) - 1))
        station  = STATIONS[safe_idx]
        url      = station.get("url", "").strip()
        if not url:
            stream_status = "BAD URL"
            return

        # Auto-apply Arctic/Nordic preset for polar territories when not in
        # manual mode — works for custom-loaded stations too
        _ARCTIC_COUNTRIES = {
            "Norway", "Sweden", "Finland", "Iceland", "Greenland",
            "Svalbard", "Faroe Islands", "Northern Norway",
            "Northern Sweden", "Northern Finland", "Jan Mayen",
        }
        stn_country = station.get("country", current_country)
        if not in_manual_mode and stn_country in _ARCTIC_COUNTRIES:
            if "Arctic/Nordic" in AUTO_PRESETS:
                globals()["active_mode"]      = "Arctic/Nordic"
                globals()["_auto_preset_idx"] = (
                    AUTO_PRESET_NAMES.index("Arctic/Nordic")
                    if "Arctic/Nordic" in AUTO_PRESET_NAMES else 0
                )
                _activate_preset(AUTO_PRESETS["Arctic/Nordic"])
                _cur_eq_f[:] = [float(v) for v in _target_eq]
                active_eq[:] = [round(v) for v in _cur_eq_f]

        filters = build_filter_chain(active_eq)
        stream_status = "CONNECTING"
        need_restart  = False

        ffmpeg_proc = subprocess.Popen(
            [_FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
             "-reconnect", "1", "-reconnect_streamed", "1",
             "-reconnect_delay_max", "5",
             "-i", url, "-af", filters, "-f", "wav", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        mpv_proc = subprocess.Popen(
            [_MPV_BIN, "--no-video",
             f"--volume={current_vol}",
             f"--input-ipc-server={MPV_SOCKET}",
             "--cache=yes", "--cache-secs=3", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Analyzer gets raw PCM (mono, 22050 Hz)
        analyzer_proc = subprocess.Popen(
            [_FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
             "-reconnect", "1", "-reconnect_streamed", "1",
             "-reconnect_delay_max", "5",
             "-i", url,
             "-f", "s16le", "-ac", "1", "-ar", str(ANALYZER_SR), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        threading.Thread(target=_audio_bridge,   args=(my_bt,), daemon=True).start()
        threading.Thread(target=_audio_analyzer, args=(my_at,), daemon=True).start()
        threading.Thread(target=_watchdog,        args=(my_bt,), daemon=True).start()
        threading.Thread(target=_icy_fetcher,    args=(url, my_bt), daemon=True).start()
        _broadcast_intel.start(my_bt)
    finally:
        _radio_lock.release()


# ══════════════════════════════════════════════════════════════════════════════
#  WATCHDOG  — exponential back-off reconnect
# ══════════════════════════════════════════════════════════════════════════════
def _watchdog(token: int):
    global stream_status, need_restart, _reconnect_attempts
    time.sleep(4.0)
    while _bridge_token == token:
        if ((ffmpeg_proc and ffmpeg_proc.poll() is not None) or
                (mpv_proc and mpv_proc.poll() is not None)):
            if stream_status == "LIVE":
                _reconnect_attempts += 1
                delay = min(MAX_RECONNECT_DELAY,
                            2.0 ** min(_reconnect_attempts - 1, 4))
                stream_status = f"RECONNECT ({int(delay)}s)"
                time.sleep(delay)
                need_restart  = True
            break
        time.sleep(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO BRIDGE  — EQ crossfade + speech/music detection routing
# ══════════════════════════════════════════════════════════════════════════════
def _audio_bridge(token: int):
    global _cur_eq_f, active_eq, stream_status
    global need_restart, last_auto_switch
    global active_mode, _auto_detection_preset, _target_eq
    global _speech_votes, _music_votes, detection_label

    CHUNK           = 4096
    restart_pending = False
    local_ff        = ffmpeg_proc
    local_mpv       = mpv_proc
    if not local_ff or not local_mpv:
        return

    try:
        hdr = local_ff.stdout.read(44)   # WAV header
        if hdr:
            local_mpv.stdin.write(hdr)
    except Exception:
        return

    stream_status = "LIVE"

    while _bridge_token == token:
        try:    data = local_ff.stdout.read(CHUNK)
        except: break
        if not data: break

        mixed_data = _audio_mixer.mix(data)
        _recorder.feed(mixed_data)   # Speech Lab: capture post-processed audio

        try:    local_mpv.stdin.write(mixed_data)
        except: break

        # EQ crossfade
        settled = True
        for i in range(10):
            diff = _target_eq[i] - _cur_eq_f[i]
            if abs(diff) > _XFADE_THRESH:
                _cur_eq_f[i] += diff * _XFADE_STEP
                settled = False
            else:
                _cur_eq_f[i] = float(_target_eq[i])
        for i in range(10):
            active_eq[i] = round(_cur_eq_f[i])

        if restart_pending:
            if settled:
                restart_pending = False
                need_restart    = True
            continue

        now = time.time()
        if now - last_auto_switch < AUTO_COOLDOWN:
            continue

        # Auto preset switching based on speech/music votes
        if _speech_votes >= VOTE_THRESHOLD and active_mode != "News/Speech" and not in_manual_mode:
            detection_label        = "SPEECH"
            active_mode            = "News/Speech"
            _auto_detection_preset = "News/Speech"
            _target_eq[:]          = list(AUTO_PRESETS["News/Speech"])
            last_auto_switch       = now
            _speech_votes          = 0
            restart_pending        = True

        elif _music_votes >= VOTE_THRESHOLD and active_mode == "News/Speech" and not in_manual_mode:
            prev = _auto_detection_preset if _auto_detection_preset != "News/Speech" else DEFAULT_AUTO
            if prev not in AUTO_PRESETS:
                prev = DEFAULT_AUTO
            detection_label  = "MUSIC"
            active_mode      = prev
            _target_eq[:]    = list(AUTO_PRESETS[prev])
            last_auto_switch = now
            _music_votes     = 0
            restart_pending  = True

    if stream_status == "LIVE":
        stream_status = "STOPPED"


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO ANALYZER  — v6 LUFSAnalyzer + BroadcastClassifier
#  Feeds PCM chunks to BroadcastIntelligence for Tier-3 AI analysis.
# ══════════════════════════════════════════════════════════════════════════════
_lufs_analyzer   = LUFSAnalyzer(sr=ANALYZER_SR)
_bc_classifier   = BroadcastClassifier(sr=ANALYZER_SR)
_news_detector   = GlobalNewsDetector()


def _audio_analyzer(token: int):
    global band_heights, _smooth_bh, peaks
    global lufs_display, genre_label, detection_label
    global _speech_votes, _music_votes, _genre_votes

    FFT_WIN   = 8192
    HOP       = 2048
    HOP_BYTES = HOP * 2
    buf       = bytearray()
    hanning   = np.hanning(FFT_WIN).astype(np.float32)
    local_ap  = analyzer_proc
    if not local_ap:
        return

    while _analyzer_token == token:
        try:    chunk = local_ap.stdout.read(HOP_BYTES)
        except: break
        if not chunk: break

        buf.extend(chunk)

        # Feed raw PCM to the Tier-3 AI orchestrator
        _broadcast_intel.feed(chunk)

        while len(buf) >= FFT_WIN * 2:
            win_bytes = bytes(buf[:FFT_WIN * 2])
            buf = buf[HOP_BYTES:]

            samples = np.frombuffer(win_bytes, dtype=np.int16).astype(np.float32)
            if samples.size < FFT_WIN:
                continue

            windowed = samples * hanning

            # LUFS — v6 K-weighted or librosa or heuristic
            lufs_val     = _lufs_analyzer.analyze(samples)
            lufs_display = lufs_val

            # FFT spectrum
            fft   = np.abs(np.fft.rfft(windowed))
            max_f = float(np.max(fft)) + 1e-6
            fft_n = fft / max_f
            rms   = float(np.sqrt(np.mean(samples**2) + 1e-9))

            n_bins = len(fft_n)
            f_max  = float(ANALYZER_SR) / 2.0
            f_min  = 20.0
            for i in range(NUM_BANDS):
                f_lo = f_min * (f_max / f_min) ** (i       / NUM_BANDS)
                f_hi = f_min * (f_max / f_min) ** ((i + 1) / NUM_BANDS)
                b_lo = max(0,      int(f_lo / f_max * n_bins))
                b_hi = min(n_bins, int(f_hi / f_max * n_bins) + 1)
                if b_hi <= b_lo: b_hi = b_lo + 1
                seg  = fft_n[b_lo:b_hi]
                raw  = min(10.0, math.sqrt(float(np.mean(seg**2))) * 14.0) if seg.size else 0.0
                alpha = 0.70 if raw > _smooth_bh[i] else 0.40
                _smooth_bh[i]   = _smooth_bh[i] * (1.0 - alpha) + raw * alpha
                band_heights[i] = _smooth_bh[i]

            # Genre from v6 BroadcastClassifier
            genre_detect = _bc_classifier.detect_genre(fft_n, rms)
            if rms > 500:
                _genre_votes[genre_detect] = _genre_votes.get(genre_detect, 0) + 1
                for k in list(_genre_votes):
                    if k != genre_detect:
                        _genre_votes[k] = max(0, _genre_votes[k] - 1)
                if _genre_votes.get(genre_detect, 0) >= GENRE_VOTE_WIN:
                    genre_label  = genre_detect
                    _genre_votes = {}
            else:
                genre_label = "QUIET"

            # Speech / Music classification — v6 BroadcastClassifier
            cls = _bc_classifier.classify(samples)
            if cls == "speech":
                _speech_votes = min(VOTE_THRESHOLD, _speech_votes + 1)
                _music_votes  = max(0, _music_votes - 1)
            else:
                _music_votes  = min(VOTE_THRESHOLD, _music_votes + 1)
                _speech_votes = max(0, _speech_votes - 1)


# ══════════════════════════════════════════════════════════════════════════════
#  SETUP WIZARD (pre-curses)
# ══════════════════════════════════════════════════════════════════════════════
def run_setup_wizard(continent: str, country: str, country_code: str) -> List[Dict]:
    print(f"\n  ╔══ Setup: {continent} → {country} ══╗")
    print(f"  ║  Fetching from RadioBrowser API...")
    raw = fetch_stations_api(country_code, country)

    if not raw:
        # Try territory fallback label
        relay = TERRITORY_RELAY_LABEL.get(country_code)
        if relay:
            print(f"  ║  ⚠ No stations — {relay}")
        else:
            print("  ║  ❌ No stations returned. Check network.")
        print("  ╚══════════════════════════════════════════════╝\n")
        return []

    print(f"  ║  Found {len(raw)} candidates. Validating streams...\n")
    total = len(raw)

    def progress_cb(checked, tot):
        pct = int(checked / tot * 40)
        bar = "█" * pct + "░" * (40 - pct)
        sys.stdout.write(f"\r  ║  [{bar}] {checked}/{tot} ")
        sys.stdout.flush()

    validated = run_validation(raw, progress_cb=progress_cb)
    print()
    ranked    = rank_stations(validated)
    print(f"\n  ║  ✅ {len(ranked)} live stations verified.")
    print(f"  ╚══════════════════════════════════════════════╝\n")
    return ranked


# ══════════════════════════════════════════════════════════════════════════════
#  CONTINENT / COUNTRY CLI MENU (pre-curses)
# ══════════════════════════════════════════════════════════════════════════════
def _pick(prompt: str, options: List[str]) -> int:
    for i, opt in enumerate(options, 1):
        print(f"  {i:>3}. {opt}")
    while True:
        try:
            raw = input(f"\n{prompt}: ").strip()
            n   = int(raw)
            if 1 <= n <= len(options):
                return n - 1
            print(f"  ⚠️  Enter 1–{len(options)}.")
        except ValueError:
            print("  ⚠️  Please enter a number.")
        except EOFError:
            sys.exit(0)


def select_country_cli() -> Tuple[str, str, str, List[Dict]]:
    continent_names = list(CONTINENTS.keys())
    print("\n🐾  Geeky Beast Radio v6 — Select Region\n")
    print("  Continent:\n")
    ci        = _pick("Enter number", continent_names)
    continent = continent_names[ci]

    country_names = list(CONTINENTS[continent].keys())
    print(f"\n  {continent} — Country/Territory ({len(country_names)} available):\n")
    ki      = _pick("Enter number", country_names)
    country = country_names[ki]
    code    = CONTINENTS[continent][country]

    cached = _load_from_db(continent, country)
    if cached:
        print(f"\n  ✔ Loaded {len(cached)} cached stations for {country}.")
        return continent, country, code, cached

    stations = run_setup_wizard(continent, country, code)
    if stations:
        _save_to_db(continent, country, stations)
        print(f"  💾 Stations saved to DB.\n")
    else:
        _db.upsert_country(continent, country, [])
    return continent, country, code, stations


# ══════════════════════════════════════════════════════════════════════════════
#  BULK FETCH ALL  — runs inside curses, live progress
# ══════════════════════════════════════════════════════════════════════════════
def fetch_all_stations_tui(scr) -> bool:
    global current_continent, current_country

    h, w = scr.getmaxyx()
    _ensure_colors()
    C1, C2, C3, C4 = _tui_C1, _tui_C2, _tui_C3, _tui_C4

    def _d(y, x, txt, attr=0):
        try:
            if 0 <= y < h and 0 <= x < w-1:
                scr.addstr(y, x, txt[:max(0, w-x-1)], attr)
        except Exception:
            pass

    # Build work list: uncached territories
    work = []
    for cont, countries in CONTINENTS.items():
        for country, code in countries.items():
            if not _db.has_country(cont, country):
                work.append((cont, country, code))

    if not work:
        scr.erase()
        msg = " ✅ All territories already cached! "
        _d(h//2, max(0, w//2 - len(msg)//2), msg, C2 | curses.A_BOLD)
        scr.refresh(); time.sleep(2.0)
        return True

    total_work  = len(work)
    total_all   = sum(len(v) for v in CONTINENTS.values())
    done        = 0
    errors      = 0
    log_lines: List[str] = []
    cancelled   = False

    scr.nodelay(False); scr.timeout(50)

    for cont, country, code in work:
        scr.timeout(0); ch = scr.getch(); scr.timeout(50)
        if ch == 27 or ch == 24:
            cancelled = True; break

        # Draw progress
        scr.erase()
        title = " 🐾 GBRadio v6 — Fetch ALL Stations "
        _d(0, max(0, w//2 - len(title)//2), title, curses.A_REVERSE | C1)

        already = total_all - total_work
        filled  = already + done
        pct     = filled / total_all
        bar_w   = max(20, w - 20)
        n_fill  = int(pct * bar_w)
        bar     = "█" * n_fill + "░" * (bar_w - n_fill)
        _d(2, 2, "Overall Progress:", C1 | curses.A_BOLD)
        _d(3, 2, f"  {int(pct*100):3d}%  [{bar}]  {filled}/{total_all}"[:w-4], C2)
        _d(5, 2, f"Now fetching:  {cont} → {country}  [{code}]", C3 | curses.A_BOLD)
        _d(6, 2, f"Remaining:     {total_work - done}  |  Errors: {errors}", curses.A_DIM)
        _d(8, 2, "Recent:", C1)
        for li, line in enumerate(log_lines[-min(10, h-14):]):
            col = C4 if "✗" in line else C2
            _d(9 + li, 4, line[:w-6], col)
        _d(h-2, 2, " ESC / Ctrl+X = cancel ", C4)
        scr.refresh()

        # Fetch + save
        try:
            relay_lbl = TERRITORY_RELAY_LABEL.get(code, "")
            raw = fetch_stations_api(code, country)
            if raw:
                validated = run_validation(raw)
                ranked    = rank_stations(validated)
                _save_to_db(cont, country, ranked)
                tag = f" [{relay_lbl}]" if relay_lbl else ""
                log_lines.append(f"✔ {country}{tag}: {len(ranked)} stations")
            else:
                _db.upsert_country(cont, country, [])
                log_lines.append(f"✗ {country}: no stations" +
                                  (f" ({relay_lbl})" if relay_lbl else ""))
                errors += 1
        except Exception as e:
            log_lines.append(f"✗ {country}: error — {str(e)[:40]}")
            errors += 1

        done += 1

    # Final summary
    scr.erase()
    if cancelled:
        title = " ⚠  Fetch Cancelled "
        _d(h//2-2, max(0, w//2 - len(title)//2), title, curses.A_REVERSE | C4)
        _d(h//2, max(0, w//2 - 20),
           f"  Fetched {done}/{total_work} territories  ", C3)
    else:
        cached_n, total_n = _cached_count()
        pct_done = cached_n / total_n * 100
        if pct_done >= 100:
            title = " ✅ ALL TERRITORIES FETCHED — 100% COMPLETE! "
            _d(h//2-3, max(0, w//2 - len(title)//2), title, curses.A_REVERSE | C2)
            _d(h//2-1, max(0, w//2 - 26),
               "  🐾 Main Menu upgraded!  Search by Country & Genre enabled.  ",
               C2 | curses.A_BOLD)
        else:
            title = f" Fetch Complete: {pct_done:.1f}% ({cached_n}/{total_n} territories) "
            _d(h//2-2, max(0, w//2 - len(title)//2), title, curses.A_REVERSE | C3)
        _d(h//2+1, max(0, w//2-20),
           f"  Errors: {errors}  |  Successful: {done-errors}  ", curses.A_DIM)

    _d(h-2, 2, " Press any key to continue ", C1)
    scr.timeout(-1); scr.getch(); scr.timeout(33); scr.nodelay(True)
    return _is_fully_cached()


# ══════════════════════════════════════════════════════════════════════════════
#  COUNTRY SELECTOR TUI  (inside curses)
# ══════════════════════════════════════════════════════════════════════════════
def country_selector_tui(scr) -> bool:
    global STATIONS, current_continent, current_country, current_idx

    h, w = scr.getmaxyx()
    _ensure_colors()
    C1, C3, C4 = _tui_C1, _tui_C3, _tui_C4

    def _d(y, x, txt, attr=0):
        try:
            if 0 <= y < h and 0 <= x < w-1:
                scr.addstr(y, x, txt[:max(0, w-x-1)], attr)
        except Exception:
            pass

    continent_names = list(CONTINENTS.keys())
    cont_idx = 0

    # Step 1: continent
    while True:
        scr.erase()
        _d(0, max(0, w//2 - 22), " 🐾 GBRadio v6 — Select Continent ",
           curses.A_REVERSE | C1)
        _d(2, 2, "↑↓ navigate   Enter confirm   ESC cancel", curses.A_DIM)
        for i, name in enumerate(continent_names):
            cached = sum(1 for c in CONTINENTS[name]
                         if _db.has_country(name, c))
            total_c = len(CONTINENTS[name])
            tag  = f"  {name:<20} [{total_c} countries, {cached} cached]  "
            attr = curses.A_REVERSE | C1 if i == cont_idx else curses.A_NORMAL
            _d(4 + i, 4, tag, attr)
        _d(h-2, 2, " ↑↓ Move   Enter Select   ESC Cancel ", C3)
        scr.refresh()
        ch = scr.getch()
        if ch == curses.KEY_UP:    cont_idx = max(0, cont_idx - 1)
        elif ch == curses.KEY_DOWN: cont_idx = min(len(continent_names)-1, cont_idx+1)
        elif ch in (10, 13):       break
        elif ch == 27:
            scr.timeout(33); return False

    chosen_continent = continent_names[cont_idx]
    country_names    = list(CONTINENTS[chosen_continent].keys())

    # Step 2: country
    cty_idx    = 0
    scroll_off = 0
    MAX_ROWS   = h - 8

    while True:
        scr.erase()
        title = f" 🐾 {chosen_continent} ({len(country_names)} entries) "
        _d(0, max(0, w//2 - len(title)//2), title, curses.A_REVERSE | C1)
        _d(2, 2, "↑↓ / PgUp/PgDn navigate   Enter confirm   ESC back", curses.A_DIM)

        if cty_idx < scroll_off:          scroll_off = cty_idx
        elif cty_idx >= scroll_off+MAX_ROWS: scroll_off = cty_idx - MAX_ROWS + 1

        visible = country_names[scroll_off: scroll_off + MAX_ROWS]
        for row, name in enumerate(visible):
            i    = row + scroll_off
            has  = _db.has_country(chosen_continent, name)
            cnt  = len(_load_from_db(chosen_continent, name)) if has else 0
            tag  = f"✔{cnt}" if has else "new"
            attr = curses.A_REVERSE | C1 if i == cty_idx else curses.A_NORMAL
            _d(4 + row, 2, f"  {i+1:>3}. {name:<28} [{tag}]  ", attr)

        _d(h-2, 2, f" ↑↓ Move   Enter Select   ESC Back  {cty_idx+1}/{len(country_names)} ", C3)
        scr.refresh()

        ch = scr.getch()
        if ch == curses.KEY_UP:
            cty_idx = max(0, cty_idx - 1)
        elif ch == curses.KEY_DOWN:
            cty_idx = min(len(country_names)-1, cty_idx+1)
        elif ch == curses.KEY_PPAGE:
            cty_idx = max(0, cty_idx - MAX_ROWS)
        elif ch == curses.KEY_NPAGE:
            cty_idx = min(len(country_names)-1, cty_idx + MAX_ROWS)
        elif ch in (10, 13):
            break
        elif ch == 27:
            return country_selector_tui(scr)

    chosen_country = country_names[cty_idx]
    chosen_code    = CONTINENTS[chosen_continent][chosen_country]

    # Step 3: load or fetch
    cached = _load_from_db(chosen_continent, chosen_country)
    if cached:
        STATIONS[:]       = cached
        current_idx       = 0
        current_continent = chosen_continent
        current_country   = chosen_country
        scr.erase()
        msg = f" ✔ Loaded {len(cached)} stations for {chosen_country} "
        _d(h//2, max(0, w//2 - len(msg)//2), msg, C1 | curses.A_BOLD)
        scr.refresh(); time.sleep(1.2)
        scr.timeout(33)
        return True

    # Need to fetch — briefly leave curses
    curses.endwin()
    new_stations = run_setup_wizard(chosen_continent, chosen_country, chosen_code)
    if new_stations:
        _save_to_db(chosen_continent, chosen_country, new_stations)
    else:
        _db.upsert_country(chosen_continent, chosen_country, [])
    scr.refresh()
    scr.timeout(33)

    if new_stations:
        STATIONS[:]       = new_stations
        current_idx       = 0
        current_continent = chosen_continent
        current_country   = chosen_country
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH BY COUNTRY (fuzzy, scrollable)
# ══════════════════════════════════════════════════════════════════════════════
def country_search_tui(scr) -> Optional[Dict]:
    h, w = scr.getmaxyx()
    _ensure_colors()
    C1, C2, C3, C4 = _tui_C1, _tui_C2, _tui_C3, _tui_C4

    all_countries = [
        (cont, country, len(_load_from_db(cont, country)))
        for cont, countries in CONTINENTS.items()
        for country in countries
    ]
    all_countries.sort(key=lambda x: x[1])

    query  = ""
    sel    = 0
    scroll = 0

    def _d(y, x, txt, attr=0):
        try:
            if 0 <= y < h and 0 <= x < w-1:
                scr.addstr(y, x, txt[:max(0, w-x-1)], attr)
        except Exception:
            pass

    def filtered():
        q = query.lower()
        return [(c, co, n) for c, co, n in all_countries
                if not q or q in co.lower() or q in c.lower()]

    scr.timeout(-1); curses.curs_set(1)

    while True:
        results  = filtered()
        max_rows = h - 9
        if sel >= len(results):   sel = max(0, len(results)-1)
        if sel < scroll:          scroll = sel
        elif sel >= scroll+max_rows: scroll = sel - max_rows + 1

        scr.erase()
        _d(0, max(0, w//2-20), " 🔍 Search by Country  [ESC = back] ",
           curses.A_REVERSE | C1)
        _d(2, 2, f"Type to filter ({len(results)}/{len(all_countries)} shown):", C1)
        box = f" {query}_" + " " * max(0, 40 - len(query))
        _d(3, 2, box[:w-4], curses.A_REVERSE)

        for row, (cont, country, count) in enumerate(results[scroll: scroll+max_rows]):
            i    = row + scroll
            y    = 5 + row
            tag  = f"[{count} stations]" if count else "[no cache]"
            line = f"  {country:<28}  {cont:<16}  {tag}"
            attr = curses.A_REVERSE | (C2 if count else C4) if i == sel else curses.A_NORMAL
            _d(y, 2, line[:w-4], attr)

        if not results:
            _d(5, 2, "  No matches.", C4)
        _d(h-2, 2, " ↑↓ scroll   PgUp/PgDn   Enter select   ESC back ", C3)
        try: scr.move(3, 3 + len(query))
        except Exception: pass
        scr.refresh()

        ch = scr.getch()
        if ch == 27 or ch == 24:
            curses.curs_set(0); scr.timeout(33); return None
        elif ch == curses.KEY_UP:    sel = max(0, sel-1)
        elif ch == curses.KEY_DOWN:  sel = min(len(results)-1, sel+1) if results else 0
        elif ch == curses.KEY_PPAGE: sel = max(0, sel - max_rows)
        elif ch == curses.KEY_NPAGE: sel = min(len(results)-1, sel + max_rows) if results else 0
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            query = query[:-1]; sel = 0; scroll = 0
        elif 32 <= ch <= 126:
            query += chr(ch); sel = 0; scroll = 0
        elif ch in (10, 13):
            if results:
                cont, country, _ = results[sel]
                stations = _load_from_db(cont, country)
                curses.curs_set(0); scr.timeout(33)
                return {"action": "play", "stations": stations,
                        "continent": cont, "country": country}


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH BY GENRE
# ══════════════════════════════════════════════════════════════════════════════
def genre_search_tui(scr) -> Optional[Dict]:
    h, w = scr.getmaxyx()
    _ensure_colors()
    C1, C2, C3, C4 = _tui_C1, _tui_C2, _tui_C3, _tui_C4

    def _d(y, x, txt, attr=0):
        try:
            if 0 <= y < h and 0 <= x < w-1:
                scr.addstr(y, x, txt[:max(0, w-x-1)], attr)
        except Exception:
            pass

    scr.erase()
    _d(h//2, max(0, w//2-22),
       " Building genre index from cached stations... ", C3 | curses.A_BOLD)
    scr.refresh()

    all_stns  = _all_stations_flat()
    genre_map: Dict[str, List[Dict]] = {}
    for s in all_stns:
        g = (s.get("genre") or "FM Radio").strip().title() or "FM Radio"
        genre_map.setdefault(g, []).append(s)

    genres = sorted(genre_map.keys(), key=lambda g: (-len(genre_map[g]), g))
    if not genres:
        scr.erase()
        _d(h//2, max(0, w//2-14), " No stations cached yet. ", C4 | curses.A_BOLD)
        _d(h//2+2, max(0, w//2-20), " Press [A] to fetch stations first. ", C3)
        _d(h-2, 2, " Press any key to continue ", C1)
        scr.timeout(-1); scr.getch(); scr.timeout(33)
        return None

    sel = 0; scroll = 0
    MAX_ROWS = h - 7

    while True:
        if sel < scroll:              scroll = sel
        elif sel >= scroll+MAX_ROWS:  scroll = sel - MAX_ROWS + 1

        scr.erase()
        _d(0, max(0, w//2-20), " 🎵 Search by Genre  [ESC = back] ",
           curses.A_REVERSE | C1)
        _d(2, 2, f"  {len(genre_map)} genres  |  {len(all_stns)} stations indexed", curses.A_DIM)

        for row, g in enumerate(genres[scroll: scroll+MAX_ROWS]):
            i    = row + scroll
            y    = 4 + row
            cnt  = len(genre_map[g])
            line = f"  {g:<28}  [{cnt} stations]"
            attr = curses.A_REVERSE | C2 if i == sel else curses.A_NORMAL
            _d(y, 2, line[:w-4], attr)

        _d(h-2, 2, " ↑↓ scroll   PgUp/PgDn   Enter select   ESC back ", C3)
        scr.refresh()

        ch = scr.getch()
        if ch == 27 or ch == 24:
            scr.timeout(33); return None
        elif ch == curses.KEY_UP:    sel = max(0, sel-1)
        elif ch == curses.KEY_DOWN:  sel = min(len(genres)-1, sel+1)
        elif ch == curses.KEY_PPAGE: sel = max(0, sel - MAX_ROWS)
        elif ch == curses.KEY_NPAGE: sel = min(len(genres)-1, sel + MAX_ROWS)
        elif ch in (10, 13):
            g = genres[sel]
            scr.timeout(33)
            return {"action": "play_genre",
                    "stations": genre_map[g][:200],
                    "genre": g}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOLOAD SETTINGS TUI
# ══════════════════════════════════════════════════════════════════════════════
def autoload_settings_tui(scr) -> None:
    h, w = scr.getmaxyx()
    _ensure_colors()
    C1, C2, C3, C4 = _tui_C1, _tui_C2, _tui_C3, _tui_C4

    cfg   = _load_autoload()
    MENU  = [
        ("Enabled",        "enabled"),
        ("Station Name",   "station_name"),
        ("Station URL",    "station_url"),
        ("Continent",      "continent"),
        ("Country",        "country"),
        ("Default Preset", "preset"),
        ("Default Volume", "volume"),
        ("── SAVE ──",     "__save__"),
    ]
    sel = 0; flash = ""; flash_t = 0.0

    def _d(y, x, txt, attr=0):
        try:
            if 0 <= y < h and 0 <= x < w-1:
                scr.addstr(y, x, txt[:max(0, w-x-1)], attr)
        except Exception:
            pass

    scr.timeout(-1)
    while True:
        scr.erase()
        _d(0, max(0, w//2-22), " 🐾 Auto-Load Settings  [ESC = back] ",
           curses.A_REVERSE | C1)
        _d(1, 2, "Configure what loads when GBRadio starts.", curses.A_DIM)

        for i, (label, key) in enumerate(MENU):
            y      = 3 + i * 2
            is_sel = (i == sel)
            attr   = curses.A_REVERSE | C1 if is_sel else curses.A_NORMAL
            if key == "__save__":
                _d(y, 2, "  [ SAVE CONFIGURATION ]  ",
                   C2 | curses.A_BOLD | (curses.A_REVERSE if is_sel else 0))
            elif key == "enabled":
                val = "✅ ON " if cfg["enabled"] else "❌ OFF"
                _d(y, 2, f"  {label:<20} {val}{'  [Enter] toggle' if is_sel else ''}", attr)
            elif key == "preset":
                val = cfg.get("preset", "Balanced")
                _d(y, 2, f"  {label:<20} {val}{'  [←/→] cycle' if is_sel else ''}", attr)
            elif key == "volume":
                val = f"{cfg.get('volume',100)}%"
                _d(y, 2, f"  {label:<20} {val}{'  [←/→] adjust' if is_sel else ''}", attr)
            else:
                val = str(cfg.get(key, ""))[:w-32] or "(not set)"
                _d(y, 2, f"  {label:<20} {val}{'  [Enter] edit' if is_sel else ''}", attr)

        if flash and time.time() - flash_t < 1.5:
            _d(h-3, 2, flash, C2 | curses.A_BOLD)
        _d(h-2, 2, " ↑↓ navigate   Enter select   ←/→ adjust   ESC back ", C3)
        scr.refresh()

        ch  = scr.getch()
        key = MENU[sel][1]

        if ch == 27 or ch == 24: break
        elif ch == curses.KEY_UP:    sel = max(0, sel-1)
        elif ch == curses.KEY_DOWN:  sel = min(len(MENU)-1, sel+1)
        elif ch in (10, 13):
            if key == "enabled":
                cfg["enabled"] = not cfg["enabled"]
            elif key == "__save__":
                flash = " ✅ Saved! " if _save_autoload(cfg) else " ❌ Save failed! "
                flash_t = time.time()
            elif key in ("station_name", "station_url", "continent", "country"):
                py = 3 + sel * 2; px = 24
                nv = _inline_input(scr, py, px, cfg.get(key, ""), max_len=60)
                if nv is not None: cfg[key] = nv
        elif ch == curses.KEY_LEFT:
            if key == "preset":
                idx = AUTO_PRESET_NAMES.index(cfg["preset"]) if cfg["preset"] in AUTO_PRESET_NAMES else 0
                cfg["preset"] = AUTO_PRESET_NAMES[(idx-1) % len(AUTO_PRESET_NAMES)]
            elif key == "volume":
                cfg["volume"] = max(0, cfg.get("volume", 100) - 5)
        elif ch == curses.KEY_RIGHT:
            if key == "preset":
                idx = AUTO_PRESET_NAMES.index(cfg["preset"]) if cfg["preset"] in AUTO_PRESET_NAMES else 0
                cfg["preset"] = AUTO_PRESET_NAMES[(idx+1) % len(AUTO_PRESET_NAMES)]
            elif key == "volume":
                cfg["volume"] = min(130, cfg.get("volume", 100) + 5)

    scr.timeout(33)


# ══════════════════════════════════════════════════════════════════════════════
#  UPGRADED MAIN MENU  (unlocked after 100% fetch)
# ══════════════════════════════════════════════════════════════════════════════
def upgraded_main_menu_tui(scr) -> Dict:
    h, w = scr.getmaxyx()
    _ensure_colors()
    C1, C2, C3, C4 = _tui_C1, _tui_C2, _tui_C3, _tui_C4

    def _d(y, x, txt, attr=0):
        try:
            if 0 <= y < h and 0 <= x < w-1:
                scr.addstr(y, x, txt[:max(0, w-x-1)], attr)
        except Exception:
            pass

    cached_n, total_n = _cached_count()
    pct     = cached_n / total_n * 100
    db_stns = _db.total_stations()

    ITEMS = [
        ("Search by Country",  "country_search"),
        ("Search by Genre",    "genre_search"),
        ("Default Station",    "autoload_station"),
        ("Auto-Load Settings", "autoload_settings"),
        ("← Back to Player",   "back"),
    ]
    if pct < 100:
        ITEMS = [("Fetch Remaining Territories", "fetch_more")] + ITEMS

    sel = 0; scr.timeout(-1)

    while True:
        scr.erase()
        title = " 🐾 GBRadio v6 — Main Menu "
        _d(0, max(0, w//2 - len(title)//2), title, curses.A_REVERSE | C1)
        _d(2, 2, f"  Cache: {cached_n}/{total_n} territories ({pct:.0f}%)   DB: {db_stns} stations",
           curses.A_DIM)
        if pct < 100:
            _d(3, 2, "  Press [A] in player to fetch remaining territories.", curses.A_DIM)

        for i, (label, _) in enumerate(ITEMS):
            y    = 5 + i * 2
            is_s = (i == sel)
            attr = curses.A_REVERSE | (C4 if "Back" in label else C1) if is_s else curses.A_NORMAL
            _d(y, max(2, w//2 - len(label)//2 - 2), f"  {label}  ",
               attr | (curses.A_BOLD if is_s else 0))

        _d(h-2, 2, " ↑↓ navigate   Enter select   ESC / ^X back ", C3)
        scr.refresh()

        ch = scr.getch()
        if ch == 27 or ch == 24:
            scr.timeout(33); return {"action": "back"}
        elif ch == curses.KEY_UP:   sel = max(0, sel-1)
        elif ch == curses.KEY_DOWN: sel = min(len(ITEMS)-1, sel+1)
        elif ch in (10, 13):
            _, action = ITEMS[sel]
            if action == "back":
                scr.timeout(33); return {"action": "back"}
            elif action == "autoload_settings":
                autoload_settings_tui(scr)
                cached_n, total_n = _cached_count(); pct = cached_n / total_n * 100
            elif action == "country_search":
                result = country_search_tui(scr)
                if result:
                    scr.timeout(33); return result
            elif action == "genre_search":
                result = genre_search_tui(scr)
                if result:
                    scr.timeout(33); return result
            elif action == "autoload_station":
                cfg = _load_autoload()
                if cfg["enabled"] and cfg["station_url"]:
                    scr.timeout(33)
                    return {"action": "play_url",
                            "name":       cfg["station_name"] or "Default Station",
                            "url":        cfg["station_url"],
                            "continent":  cfg["continent"],
                            "country":    cfg["country"],
                            "preset":     cfg["preset"],
                            "volume":     cfg["volume"]}
                else:
                    _d(h//2, 2, "  No default station set. Opening settings...",
                       C3 | curses.A_BOLD)
                    scr.refresh(); time.sleep(1.2)
                    autoload_settings_tui(scr)
            elif action == "fetch_more":
                scr.timeout(33); return {"action": "back"}


# ══════════════════════════════════════════════════════════════════════════════
#  INLINE INPUT WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def _inline_input(scr, prompt_y: int, prompt_x: int,
                  current: str, max_len: int = 24) -> Optional[str]:
    curses.curs_set(1); scr.timeout(-1)
    text = list(current); pos = len(text)

    while True:
        h, w    = scr.getmaxyx()
        box_w   = min(max_len + 4, w - prompt_x - 2)
        display = "".join(text)[:box_w - 2]
        try:
            scr.addstr(prompt_y, prompt_x,
                       f" {display:<{box_w-2}} "[:box_w], curses.A_REVERSE)
            cur_x = prompt_x + 1 + min(pos, box_w - 3)
            if 0 <= prompt_y < h and 0 <= cur_x < w - 1:
                scr.move(prompt_y, cur_x)
        except Exception:
            pass
        scr.refresh()

        ch = scr.getch()
        if ch == 27:
            result = None; break
        elif ch in (10, 13):
            result = "".join(text).strip() or current; break
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if pos > 0: text.pop(pos - 1); pos -= 1
        elif ch == curses.KEY_LEFT:  pos = max(0, pos - 1)
        elif ch == curses.KEY_RIGHT: pos = min(len(text), pos + 1)
        elif ch == curses.KEY_DC:
            if pos < len(text): text.pop(pos)
        elif 32 <= ch <= 126:
            if len(text) < max_len:
                text.insert(pos, chr(ch)); pos += 1

    curses.curs_set(0); scr.timeout(33)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  INFO OVERLAY  [I]
# ══════════════════════════════════════════════════════════════════════════════
def draw_info_overlay(scr, C1, C3, C4, C5):
    if not STATIONS or not show_info_panel:
        return
    h, w = scr.getmaxyx()
    st   = STATIONS[current_idx]

    # v6: include AI news detection
    news_lbl, news_sc = _news_detector.detect(
        st.get("name", "") + " " + st.get("genre", ""))
    lat = st.get("lat", 0.0); lon = st.get("lon", 0.0)
    geo = f"lat {lat:.3f} lon {lon:.3f}" if (lat or lon) else "—"

    lines = [
        f" ╔══ Station Info ({'★ FAV' if _is_favorite(st) else 'not fav'}) ══╗",
        f" ║  Name:    {st.get('name','?')[:w-16]}",
        f" ║  URL:     {st.get('url','?')[:w-16]}",
        f" ║  Genre:   {st.get('genre','?')}  Codec: {st.get('codec','?')}  {st.get('bitrate',0)} kbps",
        f" ║  Votes:   {st.get('votes',0)}   Score: {st.get('score',0.0):.4f}",
        f" ║  LUFS:    {lufs_display:+.1f} dB  ({_lufs_analyzer._mode})",
        f" ║  Geo:     {geo}",
        f" ║  AI News: {news_lbl} ({news_sc}%)  [{_news_detector.mode}]",
        f" ╚{'═'*(w-4)}╝",
    ]
    oy = max(1, h // 2 - len(lines) // 2)
    for i, line in enumerate(lines):
        try:
            col = C1 if i in (0, len(lines)-1) else C3
            scr.addstr(oy + i, 2, line[:w-3], col | curses.A_BOLD)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  AI INTELLIGENCE PANEL  [K]  — shows Tier-3 broadcast analysis
# ══════════════════════════════════════════════════════════════════════════════
def draw_ai_panel(scr, C1, C3, C4, C5):
    if not show_ai_panel:
        return
    h, w = scr.getmaxyx()
    st   = _broadcast_intel.state
    run  = "🔄" if st.get("running") else "✅"

    with _np_lock:
        now_pl = now_playing_raw
        stn_nm = icy_station_name

    ts = st.get("updated_at", 0)
    age = f"{int(time.time()-ts)}s ago" if ts else "—"

    lines = [
        f" ╔══ 🧠 AI Broadcast Intelligence Panel  [K]=close ══╗",
        f" ║  Mode:      {st.get('mode','—')}  {run}  (updated {age})",
        f" ║  Station:   {(stn_nm or '—')[:w-16]}",
        f" ║  Now:       {(now_pl or '—')[:w-16]}",
        f" ║  Transcript:{(st.get('transcript','—') or '—')[:w-16]}",
        f" ║  Translated:{(st.get('translation','—') or '—')[:w-16]}",
        f" ║  Topics:    {', '.join(st.get('topics', [])) or '—'}",
        f" ║  Locations: {', '.join(st.get('locations', [])) or '—'}",
        f" ║  People:    {', '.join(st.get('people', [])) or '—'}",
        f" ║  Crisis:    {st.get('crisis_label','—')}  ({st.get('crisis_score',0)}/100)",
        f" ║  Categories:{', '.join(st.get('crisis_cats',[])) or '—'}",
        f" ╚{'═'*(w-4)}╝",
    ]
    oy = max(1, h // 2 - len(lines) // 2)
    for i, line in enumerate(lines):
        try:
            crisis_s = st.get("crisis_score", 0)
            if i == 9:  # Crisis row
                col = C4 if crisis_s >= 35 else (C3 if crisis_s >= 15 else C1)
            else:
                col = C1 if i in (0, len(lines)-1) else C3
            scr.addstr(oy + i, 2, line[:w-3], col | curses.A_BOLD)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  KEYBOARD WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def _d(scr, y, x, txt, attr=0):
    try:
        h, w = scr.getmaxyx()
        if 0 <= y < h and 0 <= x < w - 1:
            scr.addstr(y, x, txt[:max(0, w - x - 1)], attr)
    except Exception:
        pass


def draw_keyboard(scr, top_y, w, C1, C3, C4, C5):
    inner = w - 2
    if inner < 4:
        return
    _d(scr, top_y, 1, "┌" + "─" * inner + "┐", C1)
    if in_manual_mode:
        hdr = (f" MANUAL ✎ '{manual_preset.get('name','MyEQ')}'"
               f" | [W/S] gain  [←/→] band  [V] save  [P] rename  [N] auto  [ESC] back ")
    else:
        hdr = (f" [↑↓] Station  [N] Preset  [M] ManualEQ  [+-] Vol"
               f"  [A] FetchAll/Menu  [C] Country  [K] AI  [R] Rec  [L] Trans  [D] Dub  [^X] Quit ")
    _d(scr, top_y + 1, 1, "│" + hdr[:inner].center(inner) + "│", C1 | curses.A_BOLD)

    for ri, row in enumerate(KB_ROWS):
        y    = top_y + 2 + ri
        keys = [(lbl, hint) for lbl, hint in row if lbl]
        if not keys:
            _d(scr, y, 1, "│" + " " * inner + "│", C1); continue
        n    = len(keys)
        cell = inner // n
        _d(scr, y, 1, "│", C1)
        cx = 2
        for lbl, hint in keys:
            tag      = f"[{hint}]{lbl}"
            cell_txt = tag[:cell - 1].ljust(cell)
            if   lbl == "QUIT":     col = C4 | curses.A_BOLD
            elif lbl == "MANUAL":   col = (C5 if in_manual_mode else C3) | curses.A_BOLD
            elif lbl == "AI":       col = (C5 if show_ai_panel else C3) | curses.A_BOLD
            elif lbl == "COUNTRY":  col = C5 | curses.A_BOLD
            elif lbl == "FAV":      col = C3 | curses.A_BOLD
            elif lbl == "SEARCH":   col = C3
            elif lbl == "INFO":     col = C1
            elif lbl == "SPEC":     col = (C5 if show_spectrum else C4)
            elif lbl == "FETCHALL": col = C5 | curses.A_BOLD
            elif lbl in ("SAVE","NAME"): col = C3
            elif "VOL" in lbl:      col = C3
            elif "STN" in lbl:      col = C1 | curses.A_BOLD
            elif "PLAY" in lbl:     col = C1 | curses.A_BOLD
            else:                   col = C3
            _d(scr, y, cx, cell_txt, col)
            cx += cell
        _d(scr, y, w - 1, "│", C1)
    _d(scr, top_y + 2 + len(KB_ROWS), 1, "└" + "─" * inner + "┘", C1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DRAW  — full TUI frame
# ══════════════════════════════════════════════════════════════════════════════
def draw_ui(scr):
    """Main TUI render — V8 optimised.

    Performance improvements over V6/V7:
    • curses.init_pair called once per session via _ensure_colors(), not per frame
    • Dirty-flag gate: _ui_state_hash() skips redraw when nothing changed
    • Spectrum/marquee animation still redraws every frame (they always move)
    • V4-quality station list: scroll hints, codec badge, idx counter
    • Status bar shows current station name for at-a-glance confirmation
    """
    global peaks, _marquee_offset, _ui_prev_hash, _ui_prev_size, _ui_frame_count

    # ── Colour setup (once per session) ───────────────────────────────────
    _ensure_colors()
    C1, C2 = _tui_C1, _tui_C2
    C3, C4 = _tui_C3, _tui_C4
    C5     = _tui_C5

    h, w = scr.getmaxyx()
    _ui_frame_count += 1

    # ── Snapshot ICY text (needs lock; do it once, reuse below) ───────────
    with _np_lock:
        np_text = now_playing_raw
        sn_text = icy_station_name

    # ── Dirty-flag gate ────────────────────────────────────────────────────
    term_size  = (h, w)
    cur_hash   = _ui_state_hash()
    now_t      = time.time()

    # These conditions bypass the dirty flag — they animate every frame
    _spectrum_live    = show_spectrum and any(bh > 0.05 for bh in band_heights)
    _marquee_live     = bool((np_text or sn_text) and
                              len(f"♪ {np_text or sn_text}  ") > w - 4)
    _flash_live       = (now_t - save_flash < 1.5) or (now_t - fav_flash < 1.5)
    _ai_ticking       = show_ai_panel   # AI panel shows live timestamps

    needs_full_redraw = (
        cur_hash != _ui_prev_hash or
        term_size != _ui_prev_size or
        _spectrum_live or
        _marquee_live or
        _flash_live or
        _ai_ticking
    )
    if not needs_full_redraw:
        return

    _ui_prev_hash = cur_hash
    _ui_prev_size = term_size

    # ── Full frame render ──────────────────────────────────────────────────
    scr.erase()
    try: scr.attron(C1); scr.border(); scr.attroff(C1)
    except: pass

    kb_top     = h - KB_HEIGHT - 1
    status_row = kb_top - 1

    draw_keyboard(scr, kb_top, w, C1, C3, C4, C5)

    # Stream-status colour used for selected station row + status bar
    if   stream_status == "LIVE":               sc = C2 | curses.A_BOLD
    elif stream_status == "CONNECTING":         sc = C3 | curses.A_BOLD
    elif stream_status.startswith("RECONNECT"): sc = C4 | curses.A_BOLD
    else:                                        sc = C4

    # ── Title bar ──────────────────────────────────────────────────────────
    safe_idx  = max(0, min(current_idx, len(STATIONS) - 1)) if STATIONS else 0
    stn       = STATIONS[safe_idx]["name"] if STATIONS else "— No Stations —"
    fav_s     = "★" if (STATIONS and _is_favorite(STATIONS[safe_idx])) else "☆"
    pname     = _current_display_name()
    stn_count = f"#{safe_idx+1}/{len(STATIONS)}" if STATIONS else ""
    region    = (f"{current_continent}/{current_country}"
                 if current_country else "No Region")
    cached_n, total_n = _cached_count()
    cache_pct  = int(cached_n / total_n * 100) if total_n else 0
    cache_tag  = "★ALL" if cache_pct >= 100 else f"{cache_pct}%"
    ai_mode    = _broadcast_intel.state.get("mode", "off")
    ai_tag     = f"[AI:{ai_mode}]" if show_ai_panel else ""
    title = (f" 🐾 GBR v7  {fav_s} {stn} {stn_count} ✦ {region} ✦ {pname}"
             f" [{detection_label}:{genre_label}] [{stream_status}]"
             f" [cache:{cache_tag}]{ai_tag} ")
    _d(scr, 0, max(1, w // 2 - len(title) // 2), title, curses.A_REVERSE | C1)

    y = 1
    if status_row - y < 2:
        scr.refresh(); return

    _d(scr, y, max(1, w // 2 - len(DEDICATION) // 2), DEDICATION, C5 | curses.A_BOLD)
    y += 1

    # ── ICY / Now-Playing scrolling marquee ────────────────────────────────
    if np_text or sn_text:
        raw_marquee = (f"♪ {np_text}  ●  {sn_text}  "
                       if (np_text and sn_text)
                       else f"♪ {np_text or sn_text}  ")
        if len(raw_marquee) > w - 4:
            _marquee_offset = (_marquee_offset + 1) % max(1, len(raw_marquee))
            display_m = (raw_marquee * 2)[_marquee_offset: _marquee_offset + (w - 4)]
        else:
            display_m       = raw_marquee
            _marquee_offset = 0
        if y < status_row:
            _d(scr, y, 2, display_m[:w - 4], C2 | curses.A_BOLD)
            y += 1

    avail = status_row - y

    # ── Spectrum analyser ──────────────────────────────────────────────────
    if show_spectrum:
        spec_rows = min(10, max(0, avail - 9))
        if spec_rows >= 3:
            _d(scr, y, 2, "◈ SPECTRUM", C1 | curses.A_BOLD)
            y += 1
            spec_bottom = y + spec_rows - 1
            bar_w = 2; gap = 1
            total_w = NUM_BANDS * (bar_w + gap) - gap
            sx      = max(2, (w - total_w) // 2)
            for i in range(NUM_BANDS):
                if band_heights[i] > peaks[i]: peaks[i] = band_heights[i]
                else:                          peaks[i] = max(0.0, peaks[i] - 0.35)
                bh = max(0, int(band_heights[i] / 10.0 * spec_rows))
                pk = max(0, min(spec_rows - 1, int(peaks[i] / 10.0 * spec_rows)))
                cx = sx + i * (bar_w + gap)
                for row in range(bh):
                    ry = spec_bottom - row
                    if y <= ry <= spec_bottom:
                        frac = row / max(1, spec_rows - 1)
                        col  = C2 if frac < 0.5 else (C3 if frac < 0.8 else C4)
                        for bx in range(bar_w):
                            _d(scr, ry, cx + bx, "▮", col)
                prow = spec_bottom - pk
                if y <= prow <= spec_bottom:
                    _d(scr, prow, cx, "▾", C4 | curses.A_BOLD)
            y = spec_bottom + 1
            avail = status_row - y

    # ── Search bar ─────────────────────────────────────────────────────────
    if search_mode:
        hits = len(_filtered_indices)
        sq   = search_query or "(type to filter)"
        _d(scr, y, 2,
           f" 🔍 SEARCH: {sq}  [{hits} matches]  ESC=clear "[:w - 4],
           C3 | curses.A_BOLD)
        y += 1; avail = status_row - y

    # ── Station list  (V4-quality: scroll hints, codec badge, instant idx) ─
    display_indices = _filtered_indices if search_mode else list(range(len(STATIONS)))
    # Give stations at least 4 rows, at most half remaining space
    stn_rows = min(len(display_indices) + 1, max(4, avail // 2))

    if avail >= 2 and display_indices:
        _d(scr, y, 2,
           f"◈ STATIONS  [{current_country}]  {len(display_indices)}/{len(STATIONS)}",
           curses.A_BOLD | C1)
        y += 1

        try:    disp_pos = display_indices.index(current_idx)
        except: disp_pos = 0

        # Keep selected station centred in its window
        vis_start = max(0, disp_pos - stn_rows // 2)
        vis_end   = min(len(display_indices), vis_start + stn_rows)
        # Fill window to stn_rows when near the end of the list
        if vis_end - vis_start < stn_rows:
            vis_start = max(0, vis_end - stn_rows)

        # Scroll-up indicator
        if vis_start > 0 and y < status_row:
            _d(scr, y - 1, w - 9, f"▲{vis_start}", C1 | curses.A_DIM)

        for di in range(vis_start, vis_end):
            if y >= status_row: break
            i      = display_indices[di]
            st     = STATIONS[i]
            is_cur = (i == current_idx)
            attr   = (curses.A_REVERSE | sc) if is_cur else curses.A_NORMAL
            pfx    = "▶ " if is_cur else "  "
            fav    = "★" if _is_favorite(st) else " "
            br     = st.get("bitrate", 0)
            br_s   = f" {br}k" if br else ""
            # Show codec label for custom/autoload stations that have no bitrate
            codec  = st.get("codec", "")
            cx_s   = f" [{codec}]" if (codec and not br) else ""
            pinned = " 📌" if st.get("pinned") else ""
            line   = f"{pfx}{fav}#{i+1} {st['name']}{br_s}{cx_s}{pinned}"
            _d(scr, y, 2, line[:w - 4], attr)
            y += 1

        # Scroll-down indicator
        remaining = len(display_indices) - vis_end
        if remaining > 0 and y <= status_row:
            _d(scr, y, w - 9, f"▼{remaining}", C1 | curses.A_DIM)

        avail = status_row - y

    elif not display_indices:
        if not STATIONS:
            _d(scr, y, 2, "  No stations. Press [C] to select a country.",
               C4 | curses.A_BOLD)
        else:
            _d(scr, y, 2,
               f'  No results for "{search_query}". Press [B] to clear.', C4)
        y += 1; avail = status_row - y

    # ── EQ bars ────────────────────────────────────────────────────────────
    if avail >= 4:
        eq_bar_h   = min(5, avail - 2)
        eq_label_y = y
        eq_bar_bot = y + eq_bar_h
        eq_freq_y  = eq_bar_bot + 1
        band_cell  = max(3, (w - 4) // 10)
        num_eq_vis = min(10, (w - 4) // band_cell)

        if in_manual_mode:
            hdr     = (f"◈ EQ  ✎MANUAL [{EQ_LABELS[selected_band]}]"
                       f"  ←/→ select   W/S adjust")
            hdr_col = C5 | curses.A_BOLD
        else:
            hdr     = (f"◈ EQ  🔒AUTO [{active_mode}]"
                       f"  (switch to MANUAL [M] to edit)")
            hdr_col = C3
        _d(scr, eq_label_y, 2, hdr[:w - 4], hdr_col)

        for i in range(num_eq_vis):
            if i >= len(active_eq): break
            gain  = active_eq[i]
            cx    = 2 + i * band_cell
            bh    = max(0, int((gain + 15) / 30.0 * eq_bar_h + 0.5))
            bh    = min(bh, eq_bar_h)
            isel  = in_manual_mode and (i == selected_band)
            char  = "█" if isel else "▄"
            col   = ((C4 if gain < 0 else C5) | (curses.A_REVERSE if isel else 0))                     if in_manual_mode else (C4 if gain < 0 else C3)
            for b in range(bh):
                ry = eq_bar_bot - b
                if eq_label_y < ry < status_row:
                    _d(scr, ry, cx, char, col)
            if eq_freq_y < status_row:
                _d(scr, eq_freq_y, cx, EQ_LABELS[i][:band_cell],
                   C1 | (curses.A_REVERSE if isel else curses.A_DIM))

    # ── Status bar ─────────────────────────────────────────────────────────
    flash_msg = ""
    if   recording_enabled:                flash_msg = " ●REC "
    elif now_t - save_flash < 1.5:         flash_msg = " ★SAVED! "
    elif now_t - fav_flash  < 1.5:         flash_msg = " ★FAV! "
    elif now_t - record_flash < 1.5 and not recording_enabled:
                                           flash_msg = " ✓REC SAVED "

    lufs_w = max(4, min(10, w - 56))
    pct    = max(0.0, min(1.0, (lufs_display + 60) / 60))
    lufs_b = "▮" * int(pct * lufs_w) + "░" * (lufs_w - int(pct * lufs_w))
    vol_w  = max(5, min(12, w - 42))
    vfill  = int(current_vol / 130.0 * vol_w)
    vbar   = "█" * vfill + "░" * (vol_w - vfill)
    # Include current station name in status bar — useful when title bar is truncated
    cur_name = (STATIONS[safe_idx]["name"][:16] if STATIONS else "")
    status = (f" VOL {current_vol:3d}%[{vbar}]"
              f"  LUFS≈{lufs_display:+.1f}[{lufs_b}]"
              f"  {pname}  {cur_name}{flash_msg} ")
    _d(scr, status_row, 2, status[:w - 4], sc)

    # ── Overlays (drawn last so they sit on top) ───────────────────────────
    if show_info_panel:
        draw_info_overlay(scr, C1, C3, C4, C5)
    if show_ai_panel:
        draw_ai_panel(scr, C1, C3, C4, C5)

    scr.refresh()


# ══════════════════════════════════════════════════════════════════════════════
#  TUI MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

async def websocket_handler(websocket, path):
    while True:
        try:
            await websocket.send(json.dumps(_broadcast_intel.state))
            await asyncio.sleep(2)
        except Exception:
            break

def start_websocket_server():
    try:
        import websockets
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(websocket_handler, "localhost", 8765)
        loop.run_until_complete(start_server)
        loop.run_forever()
    except ImportError:
        pass
    except Exception:
        pass

def _invalidate_ui():
    """Force a full redraw on the next draw_ui() call."""
    global _ui_prev_hash
    _ui_prev_hash = None


def tui(scr):
    """Main TUI event loop — V8 optimised.

    Changes vs V6/V7:
    • 30 ms timeout (V4-parity snappiness vs 33 ms)
    • _ensure_colors() called at startup and after every sub-TUI return
      so colour pairs survive curses context resets
    • _invalidate_ui() called on every state-changing keypress so the
      dirty-flag render gate never delays visible feedback
    • current_idx clamped before every start_radio() call
    • [X] clears BOTH SQLite and JSON legacy cache for current country
    • play_url action builds a full station dict (prevents KeyError on
      custom-loaded stations missing codec/genre/score keys)
    """
    global current_idx, current_vol, selected_band, need_restart
    global active_mode, _auto_preset_idx, in_manual_mode
    global manual_preset, _target_eq, _cur_eq_f, active_eq
    global save_flash, fav_flash, recording_enabled, record_start_time, record_flash
    global show_spectrum, show_info_panel, show_ai_panel
    global search_mode, search_query, _filtered_indices
    global current_continent, current_country, STATIONS
    global _tui_colors_ready

    curses.curs_set(0)
    scr.nodelay(True)
    scr.keypad(True)
    scr.timeout(30)          # 30 ms ≈ 33 fps ceiling — snappy on Termux

    # Force colour init fresh inside this curses wrapper context
    _tui_colors_ready = False
    _ensure_colors()
    _invalidate_ui()         # guarantee first frame always draws

    _activate_preset(AUTO_PRESETS[DEFAULT_AUTO])
    _filtered_indices = list(range(len(STATIONS)))

    if STATIONS:
        current_idx = 0
        start_radio(reset_votes=True)

    while True:
        if need_restart:
            need_restart = False
            start_radio(reset_votes=False)

        draw_ui(scr)
        ch = scr.getch()
        if ch == -1:
            continue

        # ── Search mode — immediate live filter ────────────────────────────
        if search_mode:
            if ch == 27 or ch in (ord('b'), ord('B')):
                search_mode = False; search_query = ""
                _filtered_indices = list(range(len(STATIONS)))
                _invalidate_ui()
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                search_query = search_query[:-1]
                _filtered_indices = _apply_search(search_query)
                _invalidate_ui()
            elif ch in (10, 13):
                search_mode = False
                if _filtered_indices:
                    current_idx = _filtered_indices[0]
                    _invalidate_ui()
                    start_radio(reset_votes=True)
            elif 32 <= ch <= 126:
                search_query += chr(ch)
                _filtered_indices = _apply_search(search_query)
                # Live-tune to first match while typing (V4 behaviour)
                if _filtered_indices:
                    current_idx = _filtered_indices[0]
                _invalidate_ui()
            continue

        # ── [L] Language translation toggle ───────────────────────────────
        if ch in (ord('l'), ord('L')):
            pairs = ["OFF", "AR_EN", "EN_AR"]
            try:
                idx = pairs.index(_translation_engine.active_lang_pair)
            except ValueError:
                idx = 0
            _translation_engine.active_lang_pair = pairs[(idx + 1) % len(pairs)]
            if (_translation_engine.active_lang_pair != "OFF"
                    and not _translation_engine.is_loaded):
                threading.Thread(
                    target=_translation_engine.load_models, daemon=True).start()
                threading.Thread(
                    target=_tts_engine.load_models, daemon=True).start()
            _invalidate_ui()

        # ── [D] Audio dubbing mode ────────────────────────────────────────
        elif ch in (ord('d'), ord('D')):
            modes = ["ORIGINAL", "DUBBED", "MOVIE"]
            try:
                idx = modes.index(_audio_mixer.mode)
            except ValueError:
                idx = 0
            _audio_mixer.mode = modes[(idx + 1) % len(modes)]
            _invalidate_ui()

        # ── Ctrl+X / Q — quit ─────────────────────────────────────────────
        elif ch == 24 or ch == ord('q'):
            break

        # ── ESC — close any open overlay ──────────────────────────────────
        elif ch == 27:
            show_info_panel = False
            show_ai_panel   = False
            if search_mode:
                search_mode  = False; search_query = ""
                _filtered_indices = list(range(len(STATIONS)))
            _invalidate_ui()

        # ── Station navigation — immediate play on each arrow press ────────
        elif ch == curses.KEY_UP:
            if STATIONS:
                current_idx = (current_idx - 1) % len(STATIONS)
                _invalidate_ui()
                start_radio(reset_votes=True)
        elif ch == curses.KEY_DOWN:
            if STATIONS:
                current_idx = (current_idx + 1) % len(STATIONS)
                _invalidate_ui()
                start_radio(reset_votes=True)
        elif ch in (10, 13):
            if STATIONS:
                current_idx = max(0, min(current_idx, len(STATIONS) - 1))
                _invalidate_ui()
                start_radio(reset_votes=False)

        # ── Volume (IPC socket — no stream restart) ────────────────────────
        elif ch in (ord('+'), ord('=')):
            current_vol = min(130, current_vol + 5)
            _invalidate_ui()
            apply_volume_now()
        elif ch in (ord('-'), ord('_')):
            current_vol = max(0, current_vol - 5)
            _invalidate_ui()
            apply_volume_now()

        # ── [N] Cycle EQ preset ────────────────────────────────────────────
        elif ch in (ord('n'), ord('N')):
            in_manual_mode   = False
            _auto_preset_idx = (_auto_preset_idx + 1) % len(AUTO_PRESET_NAMES)
            active_mode      = AUTO_PRESET_NAMES[_auto_preset_idx]
            _activate_preset(AUTO_PRESETS[active_mode])
            _invalidate_ui()

        # ── [M] Toggle manual EQ ──────────────────────────────────────────
        elif ch in (ord('m'), ord('M')):
            in_manual_mode = not in_manual_mode
            if in_manual_mode:
                _activate_preset(manual_preset["gains"])
            else:
                _activate_preset(
                    AUTO_PRESETS.get(active_mode, AUTO_PRESETS[DEFAULT_AUTO]))
            _invalidate_ui()

        # ── [C] Country selector ───────────────────────────────────────────
        elif ch in (ord('c'), ord('C')):
            _kill_procs()
            changed = country_selector_tui(scr)
            # Colour pairs may be reset by the sub-TUI — reinitialise
            _tui_colors_ready = False
            _ensure_colors()
            _invalidate_ui()
            _filtered_indices = list(range(len(STATIONS)))
            search_mode = False; search_query = ""
            if changed and STATIONS:
                current_idx = 0
                start_radio(reset_votes=True)

        # ── [F] Favourite toggle ───────────────────────────────────────────
        elif ch in (ord('f'), ord('F')):
            if STATIONS:
                _toggle_favorite(STATIONS[
                    max(0, min(current_idx, len(STATIONS) - 1))])
                fav_flash = time.time()
                _invalidate_ui()

        # ── [B] Search mode ────────────────────────────────────────────────
        elif ch in (ord('b'), ord('B')):
            search_mode  = True; search_query = ""
            _filtered_indices = list(range(len(STATIONS)))
            _invalidate_ui()

        # ── [I] Info overlay ───────────────────────────────────────────────
        elif ch in (ord('i'), ord('I')):
            show_info_panel = not show_info_panel
            _invalidate_ui()

        # ── [K] AI intelligence panel ──────────────────────────────────────
        elif ch in (ord('k'), ord('K')):
            show_ai_panel = not show_ai_panel
            _invalidate_ui()

        # ── [T] Spectrum toggle ────────────────────────────────────────────
        elif ch in (ord('t'), ord('T')):
            show_spectrum = not show_spectrum
            _invalidate_ui()

        # ── [X] Clear cache for current country ───────────────────────────
        elif ch in (ord('x'), ord('X')):
            if current_country:
                # SQLite cache
                try:
                    _conn = sqlite3.connect(DB_FILE)
                    _conn.execute(
                        "DELETE FROM stations WHERE country=?",
                        (current_country,))
                    _conn.commit()
                    _conn.close()
                except Exception:
                    pass
                # Legacy JSON cache
                try:
                    _jf = os.path.expanduser("~/gbradio_stations.json")
                    if os.path.exists(_jf):
                        with open(_jf, "r", encoding="utf-8") as _f:
                            _jcache = json.load(_f)
                        _jkey = f"{current_continent}/{current_country}"
                        if _jkey in _jcache:
                            del _jcache[_jkey]
                            with open(_jf, "w", encoding="utf-8") as _f:
                                json.dump(_jcache, _f)
                except Exception:
                    pass
            _invalidate_ui()

        # ── [A] Fetch all / main menu ──────────────────────────────────────
        elif ch in (ord('a'), ord('A')):
            _kill_procs()
            fully = _is_fully_cached()
            if not fully:
                fully = fetch_all_stations_tui(scr)
            # Reinit colours after sub-TUI
            _tui_colors_ready = False
            _ensure_colors()
            _invalidate_ui()
            if fully:
                result = upgraded_main_menu_tui(scr)
                _tui_colors_ready = False
                _ensure_colors()
                _invalidate_ui()

                if result["action"] == "play":
                    STATIONS[:]       = result["stations"]
                    current_continent = result["continent"]
                    current_country   = result["country"]
                    current_idx       = 0
                    _filtered_indices = list(range(len(STATIONS)))
                    search_mode = False; search_query = ""
                    start_radio(reset_votes=True)

                elif result["action"] == "play_url":
                    # Build a full station dict so custom URLs never cause
                    # KeyError on missing keys (codec, genre, score, etc.)
                    stn = {
                        "name":    result.get("name", "Custom URL"),
                        "url":     result.get("url",  ""),
                        "bitrate": result.get("bitrate", 0),
                        "codec":   result.get("codec",   ""),
                        "votes":   result.get("votes",   0),
                        "genre":   result.get("genre",   ""),
                        "score":   result.get("score",   0.0),
                        "pinned":  False,
                        "lat":     result.get("lat", 0.0),
                        "lon":     result.get("lon", 0.0),
                    }
                    STATIONS[:] = [stn] + STATIONS
                    current_idx       = 0
                    current_continent = result.get("continent", current_continent)
                    current_country   = result.get("country",   current_country)
                    if result.get("preset") and result["preset"] in AUTO_PRESETS:
                        active_mode = result["preset"]
                        _activate_preset(AUTO_PRESETS[active_mode])
                    if result.get("volume") is not None:
                        current_vol = result["volume"]
                        apply_volume_now()
                    _filtered_indices = list(range(len(STATIONS)))
                    start_radio(reset_votes=True)

                elif result["action"] == "play_genre":
                    STATIONS[:]       = result["stations"]
                    current_continent = ""
                    current_country   = result.get("genre", "Genre")
                    current_idx       = 0
                    _filtered_indices = list(range(len(STATIONS)))
                    search_mode = False; search_query = ""
                    start_radio(reset_votes=True)
            else:
                if STATIONS:
                    start_radio(reset_votes=False)

        # ── Manual EQ band navigation ──────────────────────────────────────
        elif ch == curses.KEY_LEFT:
            if in_manual_mode:
                selected_band = (selected_band - 1) % 10
                _invalidate_ui()
        elif ch == curses.KEY_RIGHT:
            if in_manual_mode:
                selected_band = (selected_band + 1) % 10
                _invalidate_ui()
        elif ch in (ord('w'), ord('W')):
            if in_manual_mode:
                manual_preset["gains"][selected_band] = min(
                    15, manual_preset["gains"][selected_band] + 1)
                _activate_preset(manual_preset["gains"])
                _invalidate_ui()
        elif ch in (ord('s'), ord('S')):
            if in_manual_mode:
                manual_preset["gains"][selected_band] = max(
                    -15, manual_preset["gains"][selected_band] - 1)
                _activate_preset(manual_preset["gains"])
                _invalidate_ui()
        elif ch in (ord('v'), ord('V')):
            if in_manual_mode:
                if _save_manual():
                    save_flash = time.time()
                    _invalidate_ui()
        elif ch in (ord('p'), ord('P')):
            # ── [P] Rename Manual Preset (moved from R) ────────────────────
            if in_manual_mode:
                h2, w2   = scr.getmaxyx()
                kb_top   = h2 - KB_HEIGHT - 1
                prompt_y = kb_top - 2
                prompt_x = 2
                _invalidate_ui()
                draw_ui(scr)
                _d(scr, prompt_y, prompt_x,
                   "Rename preset (Enter=confirm  ESC=cancel): ",
                   curses.color_pair(3) | curses.A_BOLD)
                scr.refresh()
                new_name = _inline_input(scr, prompt_y, prompt_x + 44,
                                         manual_preset["name"])
                if new_name:
                    manual_preset["name"] = new_name[:24]
                    _save_manual()
                    save_flash = time.time()
                _tui_colors_ready = False
                _ensure_colors()
                _invalidate_ui()
        elif ch in (ord('r'), ord('R')):
            # ── [R] Speech Lab — Record Calibration Sample (toggle) ────────
            global recording_enabled, record_start_time, record_flash
            if not _recorder.active:
                _recorder.start()
                recording_enabled  = True
                record_start_time  = time.time()
                record_flash       = time.time()
            else:
                pcm = _recorder.stop()
                recording_enabled = False
                if pcm:
                    saved_path = save_calibration_sample(pcm)
                    save_flash = time.time()   # reuse SAVED flash indicator
                record_flash = time.time()
            _invalidate_ui()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    threading.Thread(target=start_websocket_server, daemon=True).start()
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    total_countries = sum(len(v) for v in CONTINENTS.values())

    # Startup banner + AI capability report
    print("\n" + "═" * 66)
    print("  🐾  Geeky Beast Radio  v7  —  Global FM · AI Intelligence")
    print(f"  Coverage : {total_countries} FM territories across 7 continents (incl. Arctic)")
    print(f"  DB       : {DB_FILE}")
    print(f"  Presets  : {len(AUTO_PRESETS)} EQ presets  |  [N] cycle  [M] Manual EQ")
    print("═" * 66)
    for line in _ai_capability_report():
        print(line)
    print("═" * 66)

    if not _check_requirements():
        print("\n  ⚠️  Required tools missing (see above). Install and re-run.\n")
        sys.exit(1)

    # Seed hardcoded pinned stations into DB
    _ensure_hardcoded_in_db()

    # Country selection
    continent, country, code, stations = select_country_cli()

    if not stations:
        print("\n  ⚠️  No working stations found for this region.")
        print("  Launch anyway and press [C] to select another country.\n")

    STATIONS[:]       = stations
    current_continent = continent
    current_country   = country

    print(f"\n  🐾 Launching GBRadio v8 with {len(STATIONS)} stations")
    print(f"    Region : {continent} → {country}")
    print(f"    [↑↓] Station  [+/-] Vol  [N] Preset  [M] Manual EQ")
    print(f"    [A] Fetch ALL / Main Menu  [C] Country  [K] AI Panel  [R] Record  [^X] Quit")

    cached_n, total_n = _cached_count()
    pct_cached = cached_n / total_n * 100
    print(f"    Cache  : {cached_n}/{total_n} territories ({pct_cached:.1f}%)")
    if pct_cached < 100:
        print(f"    Tip    : Press [A] to fetch all territories & unlock Search")

    if HAS_WHISPER:
        print(f"    AI     : Whisper (tiny) + CrisisDetector active  →  [K] panel")
    elif HAS_SCIPY:
        print(f"    AI     : K-weighted LUFS + keyword analysis  →  [K] panel")
    else:
        print(f"    AI     : Heuristic DSP fallback  →  [K] panel")
    print()
    time.sleep(0.4)

    # Apply auto-load settings
    al = _load_autoload()
    if al["enabled"] and al["station_url"]:
        default_stn = {
            "name":    al["station_name"] or "Default Station",
            "url":     al["station_url"],
            "bitrate": 0, "codec": "", "votes": 0,
            "genre":   "", "score": 0.0, "pinned": False,
        }
        STATIONS.insert(0, default_stn)
        print(f"  🐾 Auto-load: {default_stn['name']}")
        if al.get("preset") and al["preset"] in AUTO_PRESETS:
            active_mode = al["preset"]
        if al.get("volume") is not None:
            current_vol = al["volume"]

    try:
        curses.wrapper(tui)
    finally:
        _kill_procs()

    print("\n  Bye.\n")