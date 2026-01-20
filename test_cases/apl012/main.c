/* Case 12: Multiple .c files with mpf_mfs_open in each */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

extern int init_module(void);
extern int process_module(void);

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    
    pmf_startproc(argc, argv, NULL);
    
    /* Main file uses MAIN_FILE */
    ret = mpf_mfs_open(&fcb, NULL, APL_MAIN_FILE, 0, 0, 1);
    if (ret == -1) return(-1);
    mpf_mfs_close(&fcb);
    
    init_module();
    process_module();
    
    return(0);
}
