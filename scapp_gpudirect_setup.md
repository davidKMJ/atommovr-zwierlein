# SCAPP / GPUDirect RDMA host setup

System-level settings required on the experiment control PC for
`spcm.SCAPPTransfer` (GPU-direct RDMA generation, `awg_controller/src/scapp.py`)
to work. These are OS/driver configuration, not part of this repo's install —
nothing here is enforced by `environment.yml`.

## 1. NVIDIA driver (OS-dependent)

Install the proprietary NVIDIA driver (not `nouveau`) matching the GPU and the
CUDA/cupy version you intend to use — via the distro package manager
(`ubuntu-drivers install`, `dnf install akmod-nvidia`, etc.) or the official
`.run` installer. Verify:

```bash
nvidia-smi
```

Building the Spectrum kernel module (`spcm4`) with GPUDirect support requires
the matching NVIDIA driver *source* tree with its `Module.symvers` already
generated, since `spcm4`'s `spcm_cuda.o` links directly against the driver's
exported `nvidia_p2p_get_pages` / `nvidia_p2p_dma_map_pages` / etc. symbols
(this is the same low-level API `nv_peer_mem`/`nvidia-peermem` is built on,
but `spcm4` is its own independent client of it — **`nvidia-peermem` is not
required or used by SCAPP**):

```bash
cd /usr/src/nvidia-<version>/   # from the driver .run installer
make                            # generates Module.symvers
```

Then in `spcm4-*/m4i_krnl_linux/Makefile`, uncomment and point:
```makefile
NVIDIA_DRV_SRC := /usr/src/nvidia-<version>/
```
and rebuild/install via `spcm4-*/make_spcm4_linux_kerneldrv.sh` (run as root —
required only for this build/install step, not for running SCAPP scripts
afterwards). Confirm the symbols actually resolved:

```bash
nm /lib/modules/$(uname -r)/kernel/drivers/spcm4.ko | grep nvidia_p2p
```

## 2. Locked memory (`RLIMIT_MEMLOCK`)

**Not required for the SCAPP/GPUDirect path itself** — `SCAPPTransfer`'s
buffer lives in GPU memory (`cupy` allocation) and is pinned via the NVIDIA
driver's own `nvidia_p2p_get_pages()`, which isn't gated by the host's locked-
memory rlimit at all.

It *is* required for the plain (non-SCAPP) host-RAM `DataTransfer` path used
by the regular `spcm-examples/02_generation/*` scripts: `spcm4` pins the
host-RAM DMA buffer, and under the default `ulimit -l` (usually 64 KB) that
pinning fails and `start_buffer_transfer` → `M2CMD_DATA_WAITDMA` hangs for a
normal user. Running as `root` "fixes" this incidentally (`CAP_IPC_LOCK`
bypasses the rlimit check), but the proper fix is to raise the limit for the
normal user:

`/etc/security/limits.conf`:
```
<user>  soft  memlock  unlimited
<user>  hard  memlock  unlimited
```
(requires `pam_limits.so` in the PAM stack, and a fresh login shell to take
effect — `ulimit -l` should then print `unlimited`). Keep this set even
though it's not the SCAPP fix, since the other non-CUDA examples/tests in
this repo need it to run without `sudo`.

## 3. IOMMU passthrough (`iommu=pt`)

**This is the fix for the actual SCAPP GPUDirect RDMA hang.** Symptom: the
`for card_buffer in scapp_transfer:` loop times out on `M2CMD_DATA_WAITDMA`
indefinitely (it silently retries — see `spcm`'s `DataTransfer.__next__` —
before eventually giving up), and `dmesg` shows:
```
AMD-Vi: Event Logged [IO_PAGE_FAULT domain=0x000f ...]
```
This means the IOMMU is intercepting the Spectrum card's P2P DMA write into
GPU memory and rejecting it because that address isn't mapped in the card's
strict per-device IOMMU domain — `nvidia_p2p_get_pages()` succeeding just
means the GPU driver pinned/mapped the pages in software; it says nothing
about whether the IOMMU will let the physical transaction through.

Fix (`/etc/default/grub`, append — don't replace — the existing
`GRUB_CMDLINE_LINUX_DEFAULT` value):
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet amd_iommu=on iommu=pt"
```
```bash
sudo update-grub   # or grub2-mkconfig -o /boot/grub2/grub.cfg on RHEL-based
sudo reboot
```
`iommu=pt` keeps IOMMU groups intact (useful if anything else on the box ever
needs VFIO passthrough) but uses identity-mapped passthrough instead of
strict per-device translation, so P2P DMA to physical GPU addresses is no
longer faulted. `amd_iommu=off` also works if IOMMU isolation isn't needed on
this machine at all.

Verify after reboot:
```bash
cat /proc/cmdline               # confirm amd_iommu=on iommu=pt present
dmesg | grep -i "amd-vi"        # confirm it initialized, no page faults
```
Then re-run a SCAPP script with `dmesg -w` in parallel and confirm no more
`IO_PAGE_FAULT` events appear.

## If GPUDirect RDMA still doesn't work after all of the above

Check PCIe topology and ACS next — GPUDirect P2P also requires the GPU and
the Spectrum card to be reachable without an intervening bridge redirecting
the transaction:
```bash
lspci -tv                 # GPU and Spectrum card under the same switch/root port?
lspci -vvv -s <bdf>        # check "Access Control Services" capability on each
nvidia-smi topo -m         # PIX/PXB (same switch) is ideal; NODE/SYS crosses a NUMA node
```
