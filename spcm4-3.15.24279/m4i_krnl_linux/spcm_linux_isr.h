/* SPDX-License-Identifier: GPL-2.0 */
/*
**************************************************************************

spcm_linux_isr.h                              (c) Spectrum GmbH,  09/2006

**************************************************************************

handles the interrupts

**************************************************************************
*/

#include <linux/version.h>
#include <linux/pci.h>

#if (LINUX_VERSION_CODE < KERNEL_VERSION(4,8,0))
#   define PCI_IRQ_INTX (1 << 0)
#   define PCI_IRQ_MSI  (1 << 1)
#   define PCI_IRQ_MSIX (1 << 2)
#endif
#if defined(PCI_IRQ_LEGACY) && !defined(PCI_IRQ_INTX)
#   define PCI_IRQ_INTX PCI_IRQ_LEGACY
#endif

// connection of ISR
int nConnectISR (SPCM_ST_CARDINFO* pstCard);
void vDisConnectISR (SPCM_ST_CARDINFO* pstCard);

void spcm4_vInitUserInterrupts (SPCM_ST_CARDINFO* pstCard);
