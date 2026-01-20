/* Hard Case 1: Nested function calls with arithmetic
 * Expected: 3005
 * 
 * Logic:
 *   mode = 15 % 4 = 3
 *   base = get_base(3) = BASE_C = 3000
 *   offset = get_offset(3) = 5
 *   fileno = base + offset = 3005
 */
#include <stdio.h>
#include <pmf.h>
#include <mpf_mfs.h>

#define BASE_A  1000
#define BASE_B  2000
#define BASE_C  3000
#define BASE_D  4000

int get_base(int mode);
int get_offset(int mode);

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    int mode;
    int base;
    int offset;
    
    pmf_startproc(argc, argv, NULL);
    
    mode = 15 % 4;  /* 15 % 4 = 3 */
    
    base = get_base(mode);
    offset = get_offset(mode);
    fileno = base + offset;
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}

int get_base(int mode) {
    if (mode == 0) {
        return BASE_A;
    } else if (mode == 1) {
        return BASE_B;
    } else if (mode == 2) {
        return BASE_C - 1000;  /* 2000 */
    } else if (mode == 3) {
        return BASE_C;  /* 3000 - This branch is taken */
    } else {
        return BASE_D;
    }
}

int get_offset(int mode) {
    int result;
    
    if (mode < 2) {
        result = mode + 1;
    } else if (mode >= 2 && mode <= 3) {
        result = mode + 2;  /* 3 + 2 = 5 - This branch is taken */
    } else {
        result = 10;
    }
    
    return result;
}
