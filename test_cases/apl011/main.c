/* Case 11: Switch-case statement (edge case) */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

int get_fileno(int type);

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    int type = 2;
    
    pmf_startproc(argc, argv, NULL);
    
    fileno = get_fileno(type);
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, 1);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}

int get_fileno(int type) {
    int fileno;
    
    switch(type) {
        case 1:
            fileno = APL_FILE_TYPE1;
            break;
        case 2:
            fileno = APL_FILE_TYPE2;
            break;
        case 3:
            fileno = APL_FILE_TYPE3;
            break;
        default:
            fileno = APL_FILE_DEFAULT;
            break;
    }
    
    return fileno;
}
