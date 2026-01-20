/* process.c - Processing module */
#include <stdio.h>
#include <mpf_mfs.h>
#include "apl_in.h"

int process_module(void) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    
    /* Uses arithmetic expression */
    fileno = APL_BASE + 50;
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, 1);
    if (ret == -1) return(-1);
    
    mpf_mfs_close(&fcb);
    return(0);
}
