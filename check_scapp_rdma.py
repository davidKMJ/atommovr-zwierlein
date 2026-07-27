#!/usr/bin/env python3
"""
Diagnose whether GPU-direct RDMA (SCAPP) is actually working end to end,
independent of whether the real-time streaming loop in spcm-testing.ipynb
Section 3 can sustain full-speed generation. Those are two different
questions -- this script only answers the first one.

Checks, in order (each is independent; a later failure doesn't invalidate
earlier passes):

  1. spcm4 kernel module: loaded, and built with CUDA RDMA support linked
     in (checked via `modinfo spcm4`'s `depends:` line referencing nvidia --
     a module only picks up that dependency if NVIDIA_DRV_SRC/Module.symvers
     were wired up correctly at build time).
  2. CUDA / CuPy baseline: GPU visible, and how long a *cold* CuPy op takes,
     with no card involved at all -- this alone can exceed a short
     card.timeout() and looks identical to an RDMA failure if you don't
     separate it out.
  3. Card feature bits: read-only card.features()/ext_features() query --
     as safe as Section 1 of the notebook, no channels/amplitude touched.
  4. Minimal RDMA smoke test: set up one channel and pull exactly ONE
     buffer through spcm.SCAPPTransfer, but never call card.start() -- so
     the trigger never fires and no RF is ever generated. This isolates
     "does the RDMA pin/DMA handshake complete at all" from "can the loop
     sustain real-time streaming" (which is what Section 3's timeout is
     actually about).

Run as your normal user, not root -- this should use the same permissions
Jupyter/atommovr will use (the 99-spcm4.rules udev rule installed by
make_spcm4_linux_kerneldrv.sh is what grants a normal user access to
/dev/spcm0). Safe to run with an amplifier connected: steps 1-3 never touch
the card's output stage, and step 4 never starts the trigger engine.
"""

import platform
import subprocess
import time

RESULTS = []  # (name, "PASS"|"FAIL"|"SKIP", detail)


def report(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=10)


print(f"Kernel: {platform.release()}\n")

# ---------------------------------------------------------------------------
# 1. spcm4 kernel module: loaded + built with CUDA RDMA support
# ---------------------------------------------------------------------------
print("--- 1. spcm4 kernel module ---")
try:
    lsmod = run(["lsmod"]).stdout
    if "spcm4" not in lsmod:
        report("spcm4 loaded", "FAIL", "not in `lsmod` output -- driver isn't loaded")
    else:
        report("spcm4 loaded", "PASS")

        modinfo = run(["modinfo", "spcm4"]).stdout
        depends_line = next(
            (l for l in modinfo.splitlines() if l.startswith("depends:")), ""
        )
        print(f"  {depends_line}")
        if "nvidia" in depends_line:
            report(
                "spcm4 built with CUDA RDMA",
                "PASS",
                "depends: references nvidia -- linked against nvidia.ko symbols",
            )
        else:
            report(
                "spcm4 built with CUDA RDMA",
                "FAIL",
                "depends: does not mention nvidia -- rebuilt without NVIDIA_DRV_SRC set?",
            )
except FileNotFoundError as exc:
    report("spcm4 kernel module check", "SKIP", str(exc))

try:
    dmesg_lines = [l for l in run(["dmesg"]).stdout.splitlines() if "spcm4" in l.lower()]
    print("  Recent spcm4 dmesg lines (eyeball for load-time warnings):")
    for l in dmesg_lines[-10:]:
        print(f"    {l}")
    if not dmesg_lines:
        print("    (none found -- try `sudo dmesg | grep spcm4` if this looks wrong)")
except Exception as exc:
    print(f"  (couldn't read dmesg: {exc} -- try `sudo dmesg | grep spcm4` manually)")

# ---------------------------------------------------------------------------
# 2. CUDA / CuPy baseline (no card involved at all)
# ---------------------------------------------------------------------------
print("\n--- 2. CUDA / CuPy baseline (no card involved) ---")
try:
    nvidia_smi = run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    )
    report("nvidia-smi", "PASS", nvidia_smi.stdout.strip())
except Exception as exc:
    report("nvidia-smi", "FAIL", str(exc))

try:
    import cupy as cp

    t0 = time.monotonic()
    cp.cuda.runtime.getDeviceCount()
    t_ctx = time.monotonic() - t0

    t0 = time.monotonic()
    cp.sin(cp.arange(1_000_000, dtype=cp.float64)).sum()
    cp.cuda.Stream.null.synchronize()
    t_warmup = time.monotonic() - t0

    report("CuPy device context", "PASS", f"{t_ctx:.2f}s")
    report(
        "CuPy first-op warm-up",
        "PASS" if t_warmup < 5.0 else "FAIL",
        f"{t_warmup:.2f}s"
        + (
            " -- exceeds a 5s card.timeout(); this alone could trip Section 3"
            if t_warmup >= 5.0
            else ""
        ),
    )
except Exception as exc:
    report("CuPy baseline", "FAIL", str(exc))

# ---------------------------------------------------------------------------
# 3 & 4. Card feature bits + minimal RDMA smoke test
# ---------------------------------------------------------------------------
print("\n--- 3. Card feature bits (read-only, safe with amplifier connected) ---")
try:
    import spcm
    from spcm import units

    with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:
        features = card.features()
        ext_features = card.ext_features()
        print(f"  features()     = 0x{features:08X}")
        print(f"  ext_features() = 0x{ext_features:08X}")

        # Don't hardcode a guessed constant name for the SCAPP/RDMA bit --
        # introspect whatever this installed spcm version actually exports
        # and decode the bitmasks against every match, so a wrong guess
        # can't silently produce a false PASS/FAIL.
        candidates = [
            n
            for n in dir(spcm)
            if any(k in n.upper() for k in ("SCAPP", "RDMA", "CUDA", "GPU"))
        ]
        if not candidates:
            report(
                "SCAPP/RDMA feature bit",
                "SKIP",
                "no SCAPP/RDMA/CUDA/GPU-named constant in this spcm version -- "
                "compare the raw hex above against your card's SPC_PCIEXTFEATURES docs",
            )
        else:
            for name in candidates:
                value = getattr(spcm, name)
                if not isinstance(value, int):
                    continue
                hit = bool((features | ext_features) & value)
                report(f"feature bit {name}", "PASS" if hit else "FAIL", f"0x{value:08X}")

        # ------------------------------------------------------------
        # 4. Minimal RDMA smoke test -- never calls card.start(), so the
        # trigger never fires and no RF is ever generated.
        # ------------------------------------------------------------
        print("\n--- 4. Minimal RDMA smoke test (trigger never fires, no RF output) ---")
        card.card_mode(spcm.SPC_REP_FIFO_SINGLE)
        card.timeout(15 * units.s)  # generous: allow one-time CUDA/RDMA warm-up

        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

        channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
        channels.enable(True)
        channels.output_load(50 * units.ohm)
        channels.amp(1.0 * units.V)

        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL)
        clock.sample_rate(max=True)

        scapp_transfer = spcm.SCAPPTransfer(card, direction=spcm.Direction.Generation)
        scapp_transfer.notify_samples(64 * 1024)
        scapp_transfer.allocate_buffer(1024 * 1024)

        try:
            scapp_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
            t0 = time.monotonic()
            first_buffer = next(iter(scapp_transfer))
            elapsed = time.monotonic() - t0
            report(
                "RDMA first-buffer handshake",
                "PASS",
                f"got a {first_buffer.shape} GPU buffer in {elapsed:.2f}s",
            )
        except spcm.SpcmTimeout:
            report(
                "RDMA first-buffer handshake",
                "FAIL",
                "timed out waiting for the first GPU buffer -- the RDMA path itself is "
                "not completing, independent of any real-time throughput question",
            )
        finally:
            card.stop(spcm.M2CMD_DATA_STOPDMA | spcm.M2CMD_CARD_STOP)

except Exception as exc:
    report("Card / RDMA smoke test", "FAIL", f"{type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
for name, status, _ in RESULTS:
    print(f"[{status}] {name}")
