/* mpf_mfs.h - Middleware File System Header */
#ifndef MPF_MFS_H
#define MPF_MFS_H

/* File Control Block */
typedef struct {
    int fd;
    int mode;
    int fileno;
} MPF_MFS_FCB;

/* Lock modes */
#define MPF_MFS_READLOCK  1
#define MPF_MFS_WRITELOCK 2

/* Function prototypes */
int mpf_mfs_open(MPF_MFS_FCB *fcb, void *reserved, int fileno, int mode1, int mode2, int lock);
int mpf_mfs_close(MPF_MFS_FCB *fcb);
int mpf_mfs_writerecm(MPF_MFS_FCB *fcb, int offset, int size, void *data);

#endif /* MPF_MFS_H */
