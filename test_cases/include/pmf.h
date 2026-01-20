/* pmf.h - Process Management Framework Header */
#ifndef PMF_H
#define PMF_H

/* Event head structure */
typedef struct {
    int from;
    int fidx;
} PMF_EVNHEAD;

/* Function prototypes */
int pmf_startproc(int argc, char *argv[], void *reserved);
const char* pmf_getprgname(void);
int pmf_getpidx(void);
int pmf_addevent(int eventno, void (*handler)(PMF_EVNHEAD*, void*), int size);
int pmf_mainloop(void);

#endif /* PMF_H */
