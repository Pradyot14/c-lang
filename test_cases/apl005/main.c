/* Case 5: No mpf_mfs_open - Expected: none */
#include <stdio.h>
#include <pmf.h>

int main(int argc, char *argv[]) {
    
    pmf_startproc(argc, argv, NULL);
    
    printf("Hello World!\n");
    
    return(0);
}
