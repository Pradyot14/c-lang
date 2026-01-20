/* Hard Case 3: Bitwise operations
 * Expected: 2052
 * 
 * Logic:
 *   a = 1 << 11 = 2048  (bit shift left)
 *   b = 16 >> 2 = 4     (bit shift right)
 *   fileno = a | b = 2048 | 4 = 2052  (bitwise OR)
 *
 * Binary:
 *   2048 = 100000000000
 *      4 = 000000000100
 *   2052 = 100000000100
 */
#include <stdio.h>
#include <pmf.h>
#include <mpf_mfs.h>

#define SHIFT_AMT  11
#define DIVISOR    2

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    int a, b;
    
    pmf_startproc(argc, argv, NULL);
    
    a = 1 << SHIFT_AMT;   /* 1 << 11 = 2048 */
    b = 16 >> DIVISOR;     /* 16 >> 2 = 4 */
    
    fileno = a | b;        /* 2048 | 4 = 2052 */
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}
