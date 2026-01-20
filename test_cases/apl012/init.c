/* init.c - Initialization module */
#include <stdio.h>
#include <mpf_mfs.h>
#include "apl_in.h"

int init_module(void) {
    MPF_MFS_FCB fcb;
    int ret;
    
    /* Uses APL_INIT_FILE */
    ret = mpf_mfs_open(&fcb, NULL, APL_INIT_FILE, 0, 0, 1);
    if (ret == -1) return(-1);
    
    mpf_mfs_close(&fcb);
    return(0);
}
