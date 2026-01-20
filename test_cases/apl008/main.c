/* Case 8: Nested Macros - MACRO refers to another MACRO */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    
    pmf_startproc(argc, argv, NULL);
    
    /* APL_PRIMARY -> APL_SECONDARY -> 4001 */
    ret = mpf_mfs_open(&fcb, NULL, APL_PRIMARY, 0, 0, 1);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}
