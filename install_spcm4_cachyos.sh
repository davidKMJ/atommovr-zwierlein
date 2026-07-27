#!/bin/bash
#
# CachyOS-aware wrapper around Spectrum's spcm4 kernel driver installer.
#
# Spectrum's own make_spcm4_linux_kerneldrv.sh + m4i_krnl_linux/Makefile
# (vendored under spcm4-*/) assume two things CachyOS doesn't hand you for
# free:
#
#   1. NVIDIA_DRV_SRC -- the NVIDIA driver *source* tree containing a
#      Module.symvers that matches your currently-loaded nvidia.ko. There's
#      no single standard location for this on CachyOS (no nvidia-open-dkms
#      package is required to have a working NVIDIA driver here), so this
#      script checks a few likely places instead of assuming one.
#   2. LLVM=1 -- required if your running kernel was built with Clang
#      (several CachyOS kernel variants are). Building spcm4 with a
#      mismatched toolchain fails at the modpost/link stage.
#
# This script only automates finding/validating those two things, then
# calls Spectrum's own, UNMODIFIED make_spcm4_linux_kerneldrv.sh -- nothing
# inside spcm4-*/ is patched, so this keeps working across future Spectrum
# driver version bumps (a new spcm4-x.y.z/ drop next to this script is
# picked up automatically).
#
# IMPORTANT SCOPE: this only fixes build/load-time compatibility (getting
# spcm4.ko to compile and load with CUDA RDMA symbols correctly linked). It
# does NOT fix a WAIT_DMA runtime timeout -- that points at the PCIe
# peer-to-peer data path itself (IOMMU / ACS / topology / BIOS), which no
# install script can resolve. Run check_scapp_rdma.py after this to test
# that separately.

set -eo pipefail

error_exit() {
    echo
    echo "ERROR: $1"
    echo
    exit 1
}

[ "$(id -u)" -eq 0 ] && error_exit "Run this as your normal user, not root -- it sudos internally only for the final install step."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRIVER_DIRS=()
while IFS= read -r d; do DRIVER_DIRS+=("$d"); done < <(find "$REPO_ROOT" -maxdepth 1 -type d -name 'spcm4-*')
[ "${#DRIVER_DIRS[@]}" -ge 1 ] || error_exit "No spcm4-* driver directory found next to this script."
[ "${#DRIVER_DIRS[@]}" -eq 1 ] || error_exit "Multiple spcm4-* directories found: ${DRIVER_DIRS[*]}
Remove the one you don't want, or edit this script to pick one explicitly."
DRIVER_DIR="${DRIVER_DIRS[0]}"
echo "Driver directory: $DRIVER_DIR"

command -v gcc >/dev/null 2>&1 || error_exit "gcc not found -- install your distro's base-devel/build tools first."
[ -d "/lib/modules/$(uname -r)/build" ] || error_exit "No kernel headers found at /lib/modules/$(uname -r)/build -- install the headers package matching your exact running kernel first."

# ---------------------------------------------------------------------------
# 1. LLVM=1 detection
# ---------------------------------------------------------------------------
LLVM_ARG=""
if grep -qi "clang version" /proc/version; then
    echo "Kernel was built with Clang -- will pass LLVM=1."
    LLVM_ARG="LLVM=1"
else
    echo "Kernel does not report a Clang build -- using the default toolchain (gcc)."
fi

# ---------------------------------------------------------------------------
# 2. NVIDIA_DRV_SRC detection
# ---------------------------------------------------------------------------
NV_VERSION="$(modinfo nvidia 2>/dev/null | awk '/^version:/{print $2}')"
[ -n "$NV_VERSION" ] || error_exit "Could not read the loaded nvidia module's version (modinfo nvidia failed). Is the NVIDIA driver loaded (check: lsmod | grep nvidia)?"
echo "Loaded nvidia driver version: $NV_VERSION"

if [ -n "${NVIDIA_DRV_SRC:-}" ]; then
    echo "Using NVIDIA_DRV_SRC from your environment: $NVIDIA_DRV_SRC"
else
    CANDIDATES=()

    for p in "/usr/src/nvidia-$NV_VERSION" "/usr/src/nvidia-open-$NV_VERSION"; do
        [ -f "$p/Module.symvers" ] && CANDIDATES+=("$p")
    done

    if command -v dkms >/dev/null 2>&1; then
        while IFS= read -r p; do
            [ -n "$p" ] && [ -f "$p/Module.symvers" ] && CANDIDATES+=("$p")
        done < <(dkms status 2>/dev/null | grep -i nvidia | sed -E 's#^([a-zA-Z0-9_-]+)/([0-9.]+).*#/usr/src/\1-\2#' || true)
    fi

    # last resort: search common dev locations for a manually-built
    # open-gpu-kernel-modules checkout (this is how this repo's own machine
    # ended up configured -- see the conversation history for the story)
    while IFS= read -r p; do
        [ -n "$p" ] && CANDIDATES+=("$(dirname "$p")")
    done < <(find "$HOME" -maxdepth 6 -iname "Module.symvers" -path "*kernel-open*" 2>/dev/null || true)

    UNIQUE_CANDIDATES=()
    if [ "${#CANDIDATES[@]}" -gt 0 ]; then
        while IFS= read -r p; do UNIQUE_CANDIDATES+=("$p"); done < <(printf '%s\n' "${CANDIDATES[@]}" | sort -u)
    fi

    if [ "${#UNIQUE_CANDIDATES[@]}" -eq 0 ]; then
        error_exit "Couldn't find a Module.symvers matching nvidia $NV_VERSION anywhere expected.
Set NVIDIA_DRV_SRC yourself and rerun, e.g.:
  export NVIDIA_DRV_SRC=/path/to/open-gpu-kernel-modules/kernel-open/
  $0"
    elif [ "${#UNIQUE_CANDIDATES[@]}" -gt 1 ]; then
        echo "Found multiple candidates -- pick one, export NVIDIA_DRV_SRC, and rerun:"
        printf '  %s\n' "${UNIQUE_CANDIDATES[@]}"
        exit 1
    fi
    NVIDIA_DRV_SRC="${UNIQUE_CANDIDATES[0]}"
    echo "Found matching source tree: $NVIDIA_DRV_SRC"
fi

# Best-effort sanity check: does this tree actually claim to be $NV_VERSION,
# rather than just living at a path whose name says so? Not fatal if we
# can't tell (not every source layout has version.mk) -- just a warning.
if [ -f "$NVIDIA_DRV_SRC/version.mk" ] || [ -f "$NVIDIA_DRV_SRC/../version.mk" ]; then
    if ! grep -q "$NV_VERSION" "$NVIDIA_DRV_SRC/version.mk" "$NVIDIA_DRV_SRC/../version.mk" 2>/dev/null; then
        echo "WARNING: version.mk near $NVIDIA_DRV_SRC does not mention $NV_VERSION -- double-check this is really the right source tree before continuing."
    fi
else
    echo "(No version.mk found to cross-check -- trusting the path/Module.symvers match above.)"
fi

echo
echo "About to run, as root:"
echo "  sudo env NVIDIA_DRV_SRC=\"$NVIDIA_DRV_SRC/\" $LLVM_ARG ./make_spcm4_linux_kerneldrv.sh"
echo "(unmodified Spectrum script -- this wrapper only sets these two variables)"
read -rp "Proceed? [y/N] " REPLY
[[ "$REPLY" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

cd "$DRIVER_DIR"
sudo env NVIDIA_DRV_SRC="$NVIDIA_DRV_SRC/" $LLVM_ARG ./make_spcm4_linux_kerneldrv.sh

echo
echo "Build/install finished. This only covers build+load compatibility --"
echo "run check_scapp_rdma.py next to test whether RDMA actually works."
