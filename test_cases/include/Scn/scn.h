/* scn.h - Screen/Mode Change Header */
#ifndef SCN_H
#define SCN_H

/* Mode change structure */
typedef struct {
    int now_mode;
    int next_mode;
} SCN_CHGMOD;

/* Function prototypes */
int apl_initlog(const char *name, int pidx, int size, int num);
int apl_trclog(const char *fmt, ...);

#endif /* SCN_H */
