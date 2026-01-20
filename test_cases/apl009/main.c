/* Case 9: Multiple mpf_mfs_open calls in same file */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb1, fcb2, fcb3;
    int ret;
    
    pmf_startproc(argc, argv, NULL);
    
    /* First call - direct number */
    ret = mpf_mfs_open(&fcb1, NULL, 5001, 0, 0, 1);
    if (ret == -1) return(-1);
    
    /* Second call - macro */
    ret = mpf_mfs_open(&fcb2, NULL, APL_FILE_CONFIG, 0, 0, 1);
    if (ret == -1) return(-1);
    
    /* Third call - variable with arithmetic */
    int fileno = APL_FILE_BASE + 3;
    ret = mpf_mfs_open(&fcb3, NULL, fileno, 0, 0, 1);
    if (ret == -1) return(-1);
    
    mpf_mfs_close(&fcb1);
    mpf_mfs_close(&fcb2);
    mpf_mfs_close(&fcb3);
    return(0);
}
