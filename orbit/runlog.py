"""Run logging: a per-month log file, a JSON sidecar, and actionable warnings.

Design goals, in priority order:

1. A run is reconstructible from its log alone -- code version, host, command
   line, resolved configuration, and input identities.
2. Failures are diagnosable without re-running: the log is flushed line by
   line, so a job killed at minute 45 still leaves a usable record.
3. Silent wrongness is impossible. Anything that changes the numbers is
   logged; anything suboptimal raises a warning that says what it costs and
   how to fix it.
4. Two audiences, one source. The text log and the JSON sidecar are rendered
   from the same :class:`RunRecord`, so they cannot drift.

Usage::

    rec = RunRecord(command=sys.argv)
    rec.set_config("horizontal transport", "FCT", affects_results=True,
                   source="--horizontal-fct")
    rec.warn("inputs", "UMFPACK not available; using SuperLU",
             impact="slower, ~1.5-2x more memory",
             fix="conda install -c conda-forge scikit-umfpack",
             silence="set ORBIT_LU_BACKEND=superlu")
    with rec.attach(log_path):     # tees stdout to the file
        ...run...
    rec.write_json(json_path)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_RULE = "=" * 78
_THIN = "-" * 78


# ── input identity ────────────────────────────────────────────────────────

def quick_fingerprint(path: str, chunk: int = 1 << 20) -> Dict[str, Any]:
    """Cheap identity for a possibly very large file.

    Hashes the first and last ``chunk`` bytes plus the size. This catches the
    realistic failures -- wrong file, truncated download, different version --
    without spending 10-20 s hashing a 1.4 GB lookup table on every run. It
    would NOT notice an edit in the middle of the file; use
    :func:`full_fingerprint` (``--verify-inputs``) when that matters.
    """
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if st.st_size > chunk:
            fh.seek(max(0, st.st_size - chunk))
            h.update(fh.read(chunk))
    return {"path": path, "bytes": st.st_size, "mtime": int(st.st_mtime),
            "digest": h.hexdigest()[:16], "method": "quick"}


def full_fingerprint(path: str, chunk: int = 1 << 22) -> Dict[str, Any]:
    """Full SHA-256 of a file. Slow on large inputs; used by --verify-inputs."""
    st = os.stat(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return {"path": path, "bytes": st.st_size, "mtime": int(st.st_mtime),
            "digest": h.hexdigest()[:16], "method": "full"}


def fingerprint(path: str, full: bool = False) -> Dict[str, Any]:
    """Fingerprint ``path`` -- :func:`full_fingerprint` when ``full``, else
    :func:`quick_fingerprint`. An unreadable file yields an error record
    instead of raising, so recording an input can never kill a run."""
    try:
        return full_fingerprint(path) if full else quick_fingerprint(path)
    except OSError as exc:
        return {"path": path, "error": str(exc), "method": "unavailable"}


# ── tee ───────────────────────────────────────────────────────────────────

class _Tee:
    """Duplicate a stream to a file, flushing every write."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._fh.write(data)
        self._fh.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def unique_path(path: str) -> str:
    """Never clobber: return ``path``, or ``path.2``, ``path.3``, ..."""
    if not os.path.exists(path):
        return path
    n = 2
    while os.path.exists(f"{path}.{n}"):
        n += 1
    return f"{path}.{n}"


# ── the record ────────────────────────────────────────────────────────────

@dataclass
class Warning_:
    """An actionable warning: what happened, what it costs (``impact``),
    how to fix it (``fix``), and how to silence it deliberately
    (``silence``). Trailing underscore avoids shadowing the builtin."""

    category: str
    message: str
    impact: str = ""
    fix: str = ""
    silence: str = ""

    def as_dict(self):
        return {"category": self.category, "message": self.message,
                "impact": self.impact, "fix": self.fix, "silence": self.silence}


@dataclass
class RunRecord:
    """Everything recorded about one run, accumulated as the run proceeds.

    Populate with :meth:`set_config` / :meth:`set_input` / :meth:`warn`;
    tee console output to the log file with :meth:`attach`; render the
    text log from :meth:`header` / :meth:`config_block` /
    :meth:`inputs_block` / :meth:`footer`; write the JSON sidecar with
    :meth:`write_json`. Both renderings come from this one object, so the
    text log and the sidecar cannot drift.
    """

    command: List[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    config_results: Dict[str, Dict[str, str]] = field(default_factory=dict)
    config_perf: Dict[str, Dict[str, str]] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    emissions: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Warning_] = field(default_factory=list)
    status: str = "running"

    # ── population ────────────────────────────────────────────────────────

    def set_config(self, name: str, value: Any, affects_results: bool,
                   source: str = "default"):
        target = self.config_results if affects_results else self.config_perf
        target[name] = {"value": str(value), "source": source}
        # Keep the typed value for the JSON sidecar when it is JSON-native;
        # the "value" string stays the single source for text rendering.
        if value is None or isinstance(value, (bool, int, float, list, dict)):
            target[name]["raw"] = value

    def set_input(self, label: str, path: str, full: bool = False, **extra):
        rec = fingerprint(path, full=full)
        rec.update(extra)
        self.inputs[label] = rec

    def warn(self, category: str, message: str, impact: str = "",
             fix: str = "", silence: str = "", echo: bool = True):
        w = Warning_(category, message, impact, fix, silence)
        self.warnings.append(w)
        if echo:
            print(self.render_warning(w))

    # ── rendering ─────────────────────────────────────────────────────────

    @staticmethod
    def render_warning(w: Warning_) -> str:
        lines = ["", "  " + "!" * 74,
                 f"  ! WARNING [{w.category}]: {w.message}"]
        if w.impact:
            lines.append(f"  !   Impact: {w.impact}")
        if w.fix:
            lines.append(f"  !   Fix:    {w.fix}")
        if w.silence:
            lines.append(f"  !   Silence: {w.silence}")
        lines.append("  " + "!" * 74)
        return "\n".join(lines)

    def header(self, version: str) -> str:
        git = _git_describe()
        mem = _total_memory_gb()
        lines = [
            _RULE,
            f"ORBIT {version}".ljust(46)
            + time.strftime("run %Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
            _RULE,
            "  command    " + " ".join(self.command),
            f"  code       orbit {version}  (git {git})",
            f"  host       {socket.gethostname()}   {platform.system().lower()}"
            f"-{platform.machine()}   {os.cpu_count()} cores"
            + (f" / {mem:.0f} GB" if mem else ""),
            f"  python     {platform.python_version()}  ({_lib_versions()})",
            f"  run id     {self.run_id()}",
        ]
        return "\n".join(lines)

    def run_id(self) -> str:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(self.started))
        salt = hashlib.sha256(" ".join(self.command).encode()).hexdigest()[:8]
        return f"{stamp}-{salt}"

    def config_block(self) -> str:
        def table(title, d):
            out = [f"--- {title} ".ljust(78, "-")]
            if not d:
                out.append("  (none)")
            for k, v in d.items():
                src = v["source"]
                tag = "[default]" if src == "default" else f"[{src}]"
                out.append(f"  {k:<26s} {v['value']:<28s} {tag}")
            return "\n".join(out)

        return (table("Configuration: affects results", self.config_results)
                + "\n\n"
                + table("Configuration: affects performance only", self.config_perf))

    def inputs_block(self) -> str:
        out = ["--- Inputs ".ljust(78, "-")]
        for label, rec in self.inputs.items():
            if "error" in rec:
                out.append(f"  {label:<12s} {rec['path']}  [UNREADABLE: {rec['error']}]")
                continue
            if rec.get("method") == "directory":
                extra = "  ".join(f"{k} {v}" for k, v in rec.items()
                                  if k not in ("path", "method"))
                out.append(f"  {label:<12s} {rec['path']}")
                if extra:
                    out.append(f"  {'':<12s}   {extra}")
                continue
            size = rec["bytes"] / 1e9
            size_s = f"{size:.2f} GB" if size >= 1 else f"{rec['bytes'] / 1e6:.1f} MB"
            out.append(f"  {label:<12s} {rec['path']}")
            out.append(f"  {'':<12s}   {size_s}   {rec['method']}-fingerprint "
                       f"{rec['digest']}")
        if any(r.get("method") == "quick" for r in self.inputs.values()):
            out.append("  (quick fingerprints use size + first/last 1 MB; "
                       "pass --verify-inputs for full hashes)")
        return "\n".join(out)

    def warnings_block(self) -> str:
        if not self.warnings:
            return "--- Warnings ".ljust(78, "-") + "\n  none"
        out = [f"--- Warnings ({len(self.warnings)}) ".ljust(78, "-")]
        for w in self.warnings:
            out.append(f"  [{w.category}] {w.message}")
            if w.fix:
                out.append(f"      fix: {w.fix}")
        return "\n".join(out)

    def footer(self) -> str:
        elapsed = time.time() - self.started
        return "\n".join([
            self.warnings_block(),
            "",
            _THIN,
            f"  status {self.status.upper()}   elapsed {elapsed / 60:.1f} min"
            + (f"   peak RSS {self.resources['peak_rss_mb']:.0f} MB"
               if "peak_rss_mb" in self.resources else ""),
            f"  run id {self.run_id()}",
            _THIN,
        ])

    # ── sidecar ───────────────────────────────────────────────────────────

    def as_dict(self, version: str = "") -> Dict[str, Any]:
        return {
            "run_id": self.run_id(),
            "orbit_version": version,
            "git": _git_describe(),
            "status": self.status,
            "started": self.started,
            "elapsed_s": time.time() - self.started,
            "command": self.command,
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "config": {"results_affecting": self.config_results,
                       "performance_only": self.config_perf},
            "inputs": self.inputs,
            "emissions": self.emissions,
            "results": self.results,
            "resources": self.resources,
            "warnings": [w.as_dict() for w in self.warnings],
        }

    def write_json(self, path: str, version: str = ""):
        with open(path, "w") as fh:
            json.dump(self.as_dict(version), fh, indent=2, default=str)

    # ── tee context ───────────────────────────────────────────────────────

    @contextlib.contextmanager
    def attach(self, log_path: Optional[str], preamble: Optional[str] = None):
        """Tee stdout/stderr into ``log_path`` for the duration.

        ``preamble`` is written to the file only, before the tee starts, so
        the log file can stand alone (header, config, inputs) without the
        same block being printed to the console a second time.
        """
        if not log_path:
            yield None
            return
        path = unique_path(log_path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fh = open(path, "w", buffering=1)
        if preamble:
            fh.write(preamble)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
        try:
            yield path
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            fh.close()


# ── phase-attributed RSS sampling ─────────────────────────────────────────

def _read_vmrss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


class RssSampler:
    """Attribute peak RSS to solve phases via a 1 Hz sampler thread.

    ``set_phase(name)`` labels the current phase; the sampler records the
    maximum VmRSS seen under each label (plus an explicit sample at every
    phase boundary, so short phases are never missed entirely). Costs one
    /proc read per second. Disable with ORBIT_RSS_SAMPLER=0. A 1 Hz
    cadence can miss sub-second spikes inside a phase; ru_maxrss remains
    the authoritative whole-run peak.
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.enabled = (os.environ.get("ORBIT_RSS_SAMPLER", "1") != "0"
                        and _read_vmrss_mb() is not None)
        self._phase = "startup"
        self._peaks: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self):
        if not self.enabled:
            return
        rss = _read_vmrss_mb()
        if rss is None:
            return
        with self._lock:
            if rss > self._peaks.get(self._phase, 0.0):
                self._peaks[self._phase] = rss

    def set_phase(self, name: str):
        if not self.enabled:
            return
        self._sample()          # close out the old phase
        with self._lock:
            self._phase = name
        self._sample()          # open the new one

    def start(self):
        if not self.enabled or self._thread is not None:
            return

        def _loop():
            while not self._stop.wait(self.interval_s):
                self._sample()

        self._thread = threading.Thread(target=_loop, name="rss-sampler",
                                        daemon=True)
        self._thread.start()

    def peaks(self) -> Dict[str, float]:
        """{phase: peak MB} in first-seen order, without stopping."""
        self._sample()
        with self._lock:
            return {k: round(v, 1) for k, v in self._peaks.items()}

    def stop(self) -> Dict[str, float]:
        """Stop sampling; return {phase: peak MB} in first-seen order."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2 * self.interval_s)
            self._thread = None
        return self.peaks()

    def render(self) -> str:
        with self._lock:
            peaks = dict(self._peaks)
        if not peaks:
            return ""
        return "  Peak RSS by phase: " + " | ".join(
            f"{k} {v / 1024:.1f}G" for k, v in peaks.items())


# ── environment probes ────────────────────────────────────────────────────

def _git_describe() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5).stdout.strip()
        if not sha:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, timeout=5).stdout.strip()
        return f"{sha}, {'dirty' if dirty else 'clean'}"
    except Exception:
        return "unknown"


def _lib_versions() -> str:
    out = []
    for mod in ("numpy", "scipy", "netCDF4", "numba"):
        try:
            out.append(f"{mod} {__import__(mod).__version__}")
        except Exception:
            out.append(f"{mod} -")
    return ", ".join(out)


def _total_memory_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        return None
    return None
