/* Case 6: Arithmetic expression - Expected: 2001 */
#include <stdio.h>
#include <pmf.h>
#include <mpf_mfs.h>
#include "apl_in.h"

int apl006_getfileno(int);

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    int type;
    
    pmf_startproc(argc, argv, NULL);
    
    type = 102 - 101; 
    
    fileno = apl006_getfileno(type);
    printf("fileno=[%d]\n", fileno);
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}

int apl006_getfileno(int type) {
    int fileno;
    
    if (type == 1) {
        fileno = APL_FILENO_DATA3 - 2;
    } else if (type == 2) {
        fileno = APL_FILENO_DATA3 - 1;
    } else {
        fileno = APL_FILENO_DATA3;
    }
    
    return fileno;
}
