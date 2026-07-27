// SPDX-License-Identifier: GPL-2.0
#include "spcm_cuda.h"
#ifdef WINVER
    // not supported by nvidia
#else
#   include "../m4i_krnl_linux/spcm_linux_debug.h"
#   include "../m4i_krnl_linux/spcm_linux_card.h"
#endif

void vCudaCallback (void* pvData)
    {
    SPCM_ST_CARDINFO* pstCard = pvData;
    if (pstCard->pstCudaDmaMap)
        {
        nvidia_p2p_free_dma_mapping (pstCard->pstCudaDmaMap);
        pstCard->pstCudaDmaMap = NULL;
        } 
    if (pstCard->pstCudaPageTable)
        {
        nvidia_p2p_free_page_table (pstCard->pstCudaPageTable);
        pstCard->pstCudaPageTable = NULL;
        }
    }

