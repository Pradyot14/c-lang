/* Case 2: Macro defined in same .c file - Expected: 2002 */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>

#define APL_FILENO_DATA2 (2002)

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    
    pmf_startproc(argc, argv, NULL);
    
    ret = mpf_mfs_open(&fcb, NULL, APL_FILENO_DATA2, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}
