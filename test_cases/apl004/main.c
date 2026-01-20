/* Case 4: Cross-file function - Expected: 2003 */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

/* NOTE: apl004_getmode() is declared in apl_in.h, NOT here */
/* This tests that the agent can trace through header files */

int main(int argc, char *argv[]) {
    int mode;
    
    pmf_startproc(argc, argv, NULL);
    
    mode = apl004_getmode();  /* This function is declared in apl_in.h */
    printf("mode = %d\n", mode);
    
    return(0);
}
