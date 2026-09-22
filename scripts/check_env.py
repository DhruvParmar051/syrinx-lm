"""Report the runtime environment for the experiment record.

Prints (and optionally saves) versions, hardware, backend availability, the bf16
probe, SDPA status, measured matmul TFLOPS per dtype and memory limits. Every value
comes from a library call or a measurement. Gate 0 requires this output saved to
``experiments/env.txt``.

Usage:
    uv run python scripts/check_env.py [--device mps] [--out experiments/env.txt]
"""

import argparse
import datetime as dt
import logging
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import tokenizers
import torch
import torch.nn.functional as F

from syrinx.utils.device import (
    MPS_FALLBACK_ENV_VAR,
    bytes_to_gib,
    memory_stats,
    resolve_device,
    synchronize,
)
from syrinx.utils.log import setup_logging
from syrinx.utils.precision import probe_bf16
from syrinx.utils.timing import measure_matmul_tflops

logger = logging.getLogger(__name__)

DTYPES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Namespace with ``device``, ``out``, ``matmul_n`` and ``iters``.
    """
    parser = argparse.ArgumentParser(description="Report the Syrinx runtime environment.")
    parser.add_argument("--device", help="cpu, mps or cuda (default: resolve_device())")
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    parser.add_argument("--matmul-n", type=int, default=4096, help="matmul benchmark size")
    parser.add_argument("--iters", type=int, default=30, help="timed matmul iterations")
    return parser.parse_args(argv)


def _chip_name() -> str:
    """CPU/SoC brand string (macOS sysctl), or ``platform.processor()`` elsewhere."""
    if sys.platform == "darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand.strip()
    return platform.processor() or "unknown"


def _run(cmd: list[str]) -> str | None:
    """Run a short command and return its stdout, or None if it fails."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("%s failed: %s", " ".join(cmd), exc)
        return None
    return result.stdout


def _parse_power_source(pmset_batt: str) -> str:
    """Extract the source from ``pmset -g batt`` ("AC Power", "Battery Power", ...)."""
    match = re.search(r"drawing from '([^']+)'", pmset_batt)
    return match.group(1) if match else "unknown"


def _parse_powermode(pmset_g: str) -> str:
    """Extract the active ``powermode`` value from ``pmset -g`` (raw, e.g. "1")."""
    match = re.search(r"^\s*powermode\s+(\S+)", pmset_g, flags=re.MULTILINE)
    return match.group(1) if match else "unknown"


def _power_state() -> dict[str, str]:
    """Power source and active energy-mode setting (macOS only).

    Throughput depends on both: the same matmul benchmark measured ~2-3x slower on
    battery (powermode 1) than on AC (powermode 2) on the M5 Pro. Values are raw
    ``pmset`` output; see System Settings > Battery for their meaning.
    """
    if sys.platform != "darwin":
        return {"power_source": "n/a", "pmset_powermode": "n/a"}
    batt = _run(["pmset", "-g", "batt"])
    settings = _run(["pmset", "-g"])
    return {
        "power_source": _parse_power_source(batt) if batt else "unknown",
        "pmset_powermode": _parse_powermode(settings) if settings else "unknown",
    }


def _versions() -> dict[str, str]:
    """Python, torch, numpy, tokenizers and OS versions."""
    mac_version = platform.mac_ver()[0]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
        "os": f"macOS {mac_version}" if mac_version else platform.platform(),
    }


def _sdpa_status(device: torch.device) -> str:
    """Run a tiny causal ``F.scaled_dot_product_attention`` on ``device``.

    Returns:
        ``"ok"`` or ``"FAIL: <ExceptionType>: <message>"``. Never raises.
    """
    try:
        q = torch.randn(1, 2, 16, 8, device=device)
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        synchronize(device)
        return "ok" if bool(torch.isfinite(out).all()) else "FAIL: non-finite output"
    except Exception as exc:  # noqa: BLE001 -- report, never raise
        return f"FAIL: {type(exc).__name__}: {exc}"


def _tflops_by_dtype(device: torch.device, n: int, iters: int) -> dict[str, float | str]:
    """Matmul TFLOPS for fp32, fp16 and bf16. One failing dtype doesn't stop the others.

    Returns:
        dtype name -> TFLOPS, or -> ``"FAIL: ..."`` if that dtype errored.
    """
    results: dict[str, float | str] = {}
    for name, dtype in DTYPES.items():
        try:
            results[name] = measure_matmul_tflops(device, dtype, n=n, iters=iters)
        except Exception as exc:  # noqa: BLE001 -- report, never raise
            results[name] = f"FAIL: {type(exc).__name__}: {exc}"
    return results


def collect_env_info(
    device_name: str | None, matmul_n: int = 4096, iters: int = 30
) -> dict[str, Any]:
    """Gather versions, hardware, backend status, bf16 probe, SDPA, TFLOPS and memory.

    Args:
        device_name: Passed to ``resolve_device`` (None = its default resolution).
        matmul_n: Matmul benchmark size.
        iters: Timed matmul iterations.

    Returns:
        Ordered mapping of report fields.
    """
    device = resolve_device(device_name)
    info: dict[str, Any] = dict(_versions())
    info["chip"] = _chip_name()
    info["cpu_count"] = os.cpu_count()
    info["system_memory_gib"] = round(bytes_to_gib(psutil.virtual_memory().total), 2)
    info.update(_power_state())
    info["mps_built"] = torch.backends.mps.is_built()
    info["mps_available"] = torch.backends.mps.is_available()
    info[MPS_FALLBACK_ENV_VAR] = os.environ.get(MPS_FALLBACK_ENV_VAR, "unset")
    info["device"] = device.type

    probe = probe_bf16(device)
    info["bf16_autocast_supported"] = probe.supported
    info["bf16_probe"] = probe.reason
    info["sdpa_causal"] = _sdpa_status(device)
    for name, value in _tflops_by_dtype(device, matmul_n, iters).items():
        key = f"matmul_tflops_{name} (n={matmul_n}, iters={iters})"
        info[key] = round(value, 2) if isinstance(value, float) else value

    stats = memory_stats(device)
    if stats.recommended_max is not None:
        info["recommended_max_memory_gib"] = round(bytes_to_gib(stats.recommended_max), 2)
    return info


def format_report(info: dict[str, Any], generated: dt.datetime | None = None) -> str:
    """Render ``info`` as ``key: value`` lines under a timestamped header.

    Args:
        info: Output of :func:`collect_env_info`.
        generated: Timestamp for the header (default: now, local time).

    Returns:
        The report text, ending with a newline.
    """
    stamp = (generated or dt.datetime.now().astimezone()).isoformat(timespec="seconds")
    lines = [f"# Syrinx environment report, generated {stamp}"]
    lines += [f"{key}: {value}" for key, value in info.items()]
    return "\n".join(lines) + "\n"


def write_report(text: str, out: Path) -> None:
    """Write the report to ``out`` (overwriting; git keeps the history).

    Args:
        text: Report text.
        out: Destination file; parent directories are created.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", out)


def main(argv: list[str] | None = None) -> int:
    """Entry point: collect, print and optionally save the report.

    Args:
        argv: Argument list.

    Returns:
        Process exit code (0 on success).
    """
    setup_logging()
    args = parse_args(argv)
    text = format_report(collect_env_info(args.device, args.matmul_n, args.iters))
    print(text, end="")
    if args.out is not None:
        write_report(text, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
