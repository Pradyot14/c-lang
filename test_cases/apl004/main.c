/* Case 4: Cross-file function - Expected: 2003 */
#include <stdio.h>
#include <mpf_mfs.h>
#include <pmf.h>
#include "apl_in.h"

/* Function prototype */
int apl004_getmode(void);

int main(int argc, char *argv[]) {
    int mode;
    
    pmf_startproc(argc, argv, NULL);
    
    mode = apl004_getmode();
    printf("mode = %d\n", mode);
    
    return(0);
}
