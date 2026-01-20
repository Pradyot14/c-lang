/* Hard Case 4: Multi-parameter nested conditionals
 * Expected: 5010
 * 
 * Logic:
 *   type = (7 * 3) % 5 = 21 % 5 = 1
 *   level = (20 - 8) / 4 = 12 / 4 = 3
 *   fileno = calc_fileno(type=1, level=3)
 *   
 *   In calc_fileno:
 *     type == 1, so go to first branch
 *     Inside: level == 3, so base = FILE_BASE_E = 5000
 *     offset = type * level + (level - type) = 1*3 + (3-1) = 3 + 2 = 5
 *     Wait, let me recalculate for 5010:
 *     offset = level * type + level = 3*1 + 3 = 6... no
 *     Let me set offset = (level + type) * 2 = (3+1)*2 = 8... no
 *     offset = level * 3 + type = 3*3 + 1 = 10 ✓
 *     return 5000 + 10 = 5010
 */
#include <stdio.h>
#include <pmf.h>
#include <mpf_mfs.h>

#define FILE_BASE_A  1000
#define FILE_BASE_B  2000
#define FILE_BASE_C  3000
#define FILE_BASE_D  4000
#define FILE_BASE_E  5000

int calc_fileno(int type, int level);

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    int type;
    int level;
    
    pmf_startproc(argc, argv, NULL);
    
    type = (7 * 3) % 5;    /* 21 % 5 = 1 */
    level = (20 - 8) / 4;  /* 12 / 4 = 3 */
    
    fileno = calc_fileno(type, level);
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}

int calc_fileno(int type, int level) {
    int base;
    int offset;
    
    if (type == 0) {
        if (level < 2) {
            base = FILE_BASE_A;
        } else {
            base = FILE_BASE_B;
        }
    } else if (type == 1) {
        /* type == 1: This branch is taken */
        if (level < 2) {
            base = FILE_BASE_C;
        } else if (level == 2) {
            base = FILE_BASE_D;
        } else {
            /* level == 3: This sub-branch is taken */
            base = FILE_BASE_E;  /* base = 5000 */
        }
    } else {
        base = FILE_BASE_A + FILE_BASE_B;  /* 3000 */
    }
    
    /* offset calculation */
    offset = level * 3 + type;  /* 3 * 3 + 1 = 10 */
    
    return base + offset;  /* 5000 + 10 = 5010 */
}
