/* Case 7: Callback function - Expected: 2003 */
#include <stdio.h>
#include <pmf.h>
#include <mpf_mfs.h>
#include <Scn/scn.h>
#include "apl_in.h"

void apl100d_chmod(PMF_EVNHEAD *eventhead, void *eventdata);

int main(int argc, char *argv[]) {
    pmf_startproc(argc, argv, NULL);
    
    apl_initlog(pmf_getprgname(), pmf_getpidx(), APL_LOG_SIZE, APL_LOG_NUM);
    pmf_addevent(APL_EVENTNO_1, apl100d_chmod, sizeof(SCN_CHGMOD));
    pmf_mainloop();
    
    return(0);
}

void apl100d_chmod(PMF_EVNHEAD *eventhead, void *eventdata) {
    SCN_CHGMOD *mode;
    MPF_MFS_FCB fcb;
    AplData3 data;
    int ret;
    
    mode = (SCN_CHGMOD *)eventdata;
    
    apl_trclog("from:%d%d nowmode=[%d] nextmode=[%d]",
               eventhead->from, eventhead->fidx,
               mode->now_mode, mode->next_mode);
    
    data.now_mode = mode->now_mode;
    data.next_mode = mode->next_mode;
    
    ret = mpf_mfs_open(&fcb, NULL, APL_FILENO_DATA3, 0, 0, MPF_MFS_WRITELOCK);
    if (ret == -1) {
        return;
    }
    
    mpf_mfs_writerecm(&fcb, 0, 0, &data);
    mpf_mfs_close(&fcb);
}
