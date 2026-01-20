/* Case 1: Direct numeric literal - Expected: 2001 */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    
    pmf_startproc(argc, argv, NULL);
    
    ret = mpf_mfs_open(&fcb, NULL, 2001, 0, 0, 1);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}

