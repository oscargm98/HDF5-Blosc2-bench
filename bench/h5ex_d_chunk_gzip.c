/************************************************************

  This example shows how to read and write data to a dataset
  using gzip compression (also called zlib or deflate).  The
  program first checks if gzip compression is available,
  then if it is it writes integers to a dataset using gzip,
  then closes the file.  Next, it reopens the file, reads
  back the data, and outputs the type of compression and the
  maximum value in the dataset to the screen.

 ************************************************************/

#include "hdf5.h"
#include "caterva.h"
#include "blosc2.h"
#include <stdio.h>
#include <stdlib.h>

#define DIM0            32
#define DIM1            64
#define CHUNK0          4
#define CHUNK1          8

#define FILE_CAT            "h5ex_cat.h5"
#define FILE_H5             "h5ex_h5.h5"
#define DATASET_CAT         "DSCAT"
#define DATASET_H5          "DSH5"

int
main (void)
{
    blosc_init();

    hsize_t         dims[2] = {DIM0, DIM1},
            chunk[2] = {CHUNK0, CHUNK1},
            start[2],
            stride[2],
            count[2],
            block[2];
    int             wdata[DIM0 * DIM1],          /* Write buffer */
    rdata[CHUNK0 * CHUNK1];          /* Read buffer */
    hsize_t         i, j;

    uint8_t ndim = 2;
    hsize_t *offset = malloc(8 * sizeof(int32_t));
    int64_t *chunksdim = malloc(8 * sizeof(int64_t ));
    int64_t *nchunk_ndim = malloc(8 * sizeof(int64_t ));
    for ( i = 0; i < ndim; ++i) {
        chunksdim[i] = (dims[i] - 1) / chunk[i] + 1;
    }
    int chunksize = CHUNK0 * CHUNK1;

    /*
     * Check if gzip compression is available and can be used for both
     * compression and decompression.  Normally we do not perform error
     * checking in these examples for the sake of clarity, but in this
     * case we will make an exception because this filter is an
     * optional part of the hdf5 library.
     */

    /*
     * Initialize data.
     */
    for (i=0; i< DIM0; i++)
        for (j=0; j< DIM1; j++)
            wdata[i * DIM1 + j] = i * j - j;

    hid_t           file_cat_w, file_cat_r, file_h5_w, file_h5_r, space, mem_space,
                    dset_cat_w, dset_cat_r, dset_h5_w, dset_h5_r, dcpl;    /* Handles */
    herr_t          status;
    unsigned        flt_msk = 0;

    file_cat_w = H5Fcreate (FILE_CAT, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    file_h5_w = H5Fcreate (FILE_H5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    space = H5Screate_simple (2, (const hsize_t *) dims, NULL);
    hsize_t memsize = (hsize_t) chunksize;
    mem_space = H5Screate_simple (1, &memsize, NULL);
    dcpl = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_chunk (dcpl, 2, chunk);
    dset_cat_w = H5Dcreate (file_cat_w, DATASET_CAT, H5T_STD_I32LE, space, H5P_DEFAULT, dcpl,
                          H5P_DEFAULT);
    status = H5Pset_deflate (dcpl, 1);
    dset_h5_w = H5Dcreate (file_h5_w, DATASET_H5, H5T_STD_I32LE, space, H5P_DEFAULT, dcpl,
                         H5P_DEFAULT);
    start[0] = 0;
    stride[0] = chunksize;
    count[0] = 1;
    block[0] = chunksize;
    status = H5Sselect_hyperslab (mem_space, H5S_SELECT_SET, start, stride, count, block);

    for(int nchunk = 0; nchunk < 64; nchunk++) {
        printf("\nchunk %d\n", nchunk);
        // Get chunk offset
        blosc2_unidim_to_multidim((int8_t) ndim, (int64_t *) chunksdim, nchunk, (int64_t *) nchunk_ndim);
        for (int i = 0; i < ndim; ++i) {
            offset[i] = (hsize_t) nchunk_ndim[i] * chunk[i];
        }

        // Use H5Dwrite to save caterva compressed buffer
        status = H5Dwrite_chunk(dset_cat_w, H5P_DEFAULT, flt_msk, offset, chunksize * sizeof(int),
                                 &wdata[nchunk * chunksize]);

        start[0] = nchunk_ndim[0] * CHUNK0;
        start[1] = nchunk_ndim[1] * CHUNK1;
        stride[0] = CHUNK0;
        stride[1] = CHUNK1;
        count[0] = 1;
        count[1] = 1;
        block[0] = CHUNK0;
        block[1] = CHUNK1;
        status = H5Sselect_hyperslab (space, H5S_SELECT_SET, start, stride, count,
                                      block);
        // Use H5Dwrite to compress and save buffer using gzip
        status = H5Dwrite(dset_h5_w, H5T_NATIVE_INT, mem_space, space, H5P_DEFAULT,
                          &wdata[nchunk * chunksize]);

        for (i=0; i<chunksize; i++) {
                printf(" %d", wdata[nchunk * chunksize + i]);
        }
    }

    // Close and release resources.
    status = H5Pclose (dcpl);
    status = H5Sclose (space);
    status = H5Sclose (mem_space);
    status = H5Fclose (file_cat_w);
    status = H5Fclose (file_h5_w);
    status = H5Dclose (dset_cat_w);
    status = H5Dclose (dset_h5_w);

    printf("\n\nWrite finished, let's read!\n\n");

    // Open HDF5 dataset
    file_cat_r = H5Fopen (FILE_CAT, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_cat_r = H5Dopen (file_cat_r, DATASET_CAT, H5P_DEFAULT);
    file_h5_r = H5Fopen (FILE_H5, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_h5_r = H5Dopen (file_h5_r, DATASET_H5, H5P_DEFAULT);
    space = H5Screate_simple (2, (const hsize_t *) dims, NULL);
    mem_space = H5Screate_simple (1, &memsize, NULL);
    start[0] = 0;
    stride[0] = chunksize;
    count[0] = 1;
    block[0] = chunksize;
    status = H5Sselect_hyperslab (mem_space, H5S_SELECT_SET, start, stride, count, block);

    for(int nchunk = 0; nchunk < 64; nchunk++) {
        printf("\nchunk %d\n", nchunk);
        // Get chunk offset
        blosc2_unidim_to_multidim((int8_t) ndim, (int64_t *) chunksdim, nchunk, (int64_t *) nchunk_ndim);
        for (int i = 0; i < ndim; ++i) {
            offset[i] = nchunk_ndim[i] * chunk[i];
        }

        // Read caterva compressed buffer
        status = H5Dread_chunk(dset_cat_r, H5P_DEFAULT, (const hsize_t *) offset, &flt_msk,
                               rdata);
        for (i=0; i<chunksize; i++) {
            printf(" %d", rdata[i]);
        }
        printf("\n");

        start[0] = nchunk_ndim[0] * CHUNK0;
        start[1] = nchunk_ndim[1] * CHUNK1;
        stride[0] = CHUNK0;
        stride[1] = CHUNK1;
        count[0] = 1;
        count[1] = 1;
        block[0] = CHUNK0;
        block[1] = CHUNK1;
        status = H5Sselect_hyperslab (space, H5S_SELECT_SET, start, stride, count,
                                      block);
        // Read HDF5 buffer
        status = H5Dread (dset_h5_r, H5T_NATIVE_INT, mem_space, space, H5P_DEFAULT,
                          rdata);
        for (i=0; i<chunksize; i++) {
                printf(" %d", rdata[i]);
        }

    }

    // Close and release resources.
    status = H5Sclose (space);
    status = H5Sclose (mem_space);
    status = H5Dclose (dset_cat_r);
    status = H5Fclose (file_cat_r);
    status = H5Dclose (dset_h5_r);
    status = H5Fclose (file_h5_r);

    blosc_destroy();

    return 0;
}
