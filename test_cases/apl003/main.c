/* Case 3: Variable from macro in header - Expected: 2003 */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    
    pmf_startproc(argc, argv, NULL);
    
    fileno = APL_FILENO_DATA3;
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}
