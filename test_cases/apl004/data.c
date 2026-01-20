/* data.c - Contains the function with mpf_mfs_open */
#include <stdio.h>
#include <mpf_mfs.h>
#include "apl_in.h"

int apl004_getmode(void) {
    MPF_MFS_FCB fcb;
    int ret;
    
    ret = mpf_mfs_open(&fcb, NULL, APL_FILENO_DATA3, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}
