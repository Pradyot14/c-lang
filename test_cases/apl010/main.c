/* Case 10: Chained function calls - func1 calls func2 which has mpf_mfs_open */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

int process_data(int mode);
int open_file(int fileno);

int main(int argc, char *argv[]) {
    int result;
    
    pmf_startproc(argc, argv, NULL);
    
    /* main -> process_data -> open_file -> mpf_mfs_open */
    result = process_data(1);
    
    return result;
}

int process_data(int mode) {
    int fileno;
    
    if (mode == 1) {
        fileno = APL_DATA_FILE;
    } else {
        fileno = APL_LOG_FILE;
    }
    
    return open_file(fileno);
}

int open_file(int fileno) {
    MPF_MFS_FCB fcb;
    int ret;
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, 1);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}
