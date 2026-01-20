/* Hard Case 5: Ternary operator chain
 * Expected: 3500
 * 
 * Logic:
 *   n = 17
 *   m = 17 / 5 = 3
 *   
 *   fileno = (m == 0) ? 1000 :
 *            (m == 1) ? 1500 :
 *            (m == 2) ? 2000 :
 *            (m == 3) ? 2500 + compute_bonus(n) :  <-- This branch (m=3)
 *                       4000;
 *   
 *   compute_bonus(17):
 *     17 > 15, so return HIGH_BONUS = 1000
 *   
 *   fileno = 2500 + 1000 = 3500
 */
#include <stdio.h>
#include <pmf.h>
#include <mpf_mfs.h>

#define LOW_BONUS   100
#define MED_BONUS   500
#define HIGH_BONUS  1000

int compute_bonus(int n);

int main(int argc, char *argv[]) {
    MPF_MFS_FCB fcb;
    int ret;
    int fileno;
    int n, m;
    
    pmf_startproc(argc, argv, NULL);
    
    n = 17;
    m = n / 5;  /* 17 / 5 = 3 */
    
    /* Ternary chain - m=3 takes the 4th option */
    fileno = (m == 0) ? 1000 :
             (m == 1) ? 1500 :
             (m == 2) ? 2000 :
             (m == 3) ? 2500 + compute_bonus(n) :
                        4000;
    
    ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, MPF_MFS_READLOCK);
    if (ret == -1) {
        return(-1);
    }
    
    mpf_mfs_close(&fcb);
    return(0);
}

int compute_bonus(int n) {
    if (n <= 5) {
        return LOW_BONUS;   /* 100 */
    } else if (n <= 10) {
        return MED_BONUS;   /* 500 */
    } else if (n <= 15) {
        return MED_BONUS + LOW_BONUS;  /* 600 */
    } else {
        return HIGH_BONUS;  /* n=17 > 15, so return 1000 */
    }
}
