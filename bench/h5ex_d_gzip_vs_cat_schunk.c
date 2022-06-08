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

#define FILE_CAT            "h5ex_cat.h5"
#define FILE_H5             "h5ex_h5.h5"
#define DATASET_CAT         "DSCAT"
#define DATASET_H5          "DSH5"

int comp(char* urlpath_input, char* urlpath_cat)
{
    blosc_init();

    // Parameters definition
    caterva_config_t cfg = CATERVA_CONFIG_DEFAULTS;
    caterva_ctx_t *ctx;
    caterva_ctx_new(&cfg, &ctx);
    caterva_array_t *arr;
    caterva_open(ctx, urlpath_input, &arr);

    int8_t ndim = arr->ndim;
    int64_t *shape = arr->shape;
    int64_t extshape[8];
    int32_t *chunkshape = arr->chunkshape;
    int64_t *extchunkshape = arr->extchunkshape;
    hsize_t offset[8];
    int64_t chunksdim[8];
    int64_t nchunk_ndim[8];
    hsize_t chunks[8];
    int64_t chunknelems = 1;
    for (int i = 0; i < ndim; ++i) {
        offset[i] = nchunk_ndim[i] = 0;
        chunksdim[i] = (shape[i] - 1) / chunkshape[i] + 1;
        extshape[i] = extchunkshape[i] * chunksdim[i];
        chunknelems *= extchunkshape[i];
        chunks[i] = extchunkshape[i];
    }
    blosc2_remove_urlpath(urlpath_cat);

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = BLOSC_ZLIB;
    cparams.typesize = arr->itemsize;
    cparams.clevel = 1;
    cparams.nthreads = 6;
    cparams.blocksize = arr->sc->blocksize;
    blosc2_context *cctx;
    cctx = blosc2_create_cctx(cparams);
    blosc2_storage storage = {.cparams=&cparams, .contiguous=true, .urlpath = NULL};
    blosc2_schunk* wschunk = blosc2_schunk_new(&storage);

    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    dparams.nthreads = 6;
    blosc2_context *dctx;
    dctx = blosc2_create_dctx(dparams);
    blosc2_schunk* rschunk;

    int32_t chunksize = arr->sc->chunksize;
    uint8_t *chunk = malloc(chunksize);
    uint8_t *cchunk = malloc(chunksize);
    int32_t *buffer_cat = malloc(chunksize);
    uint8_t *cbuffer = malloc(arr->sc->nbytes);
    int32_t *buffer_h5 = malloc(chunksize);

    blosc_timestamp_t t0, t1;
    double cat_time_w = 0;
    double cat_time_r = 0;
    double h5_time_w = 0;
    double h5_time_r = 0;
    int compressed, decompressed;
    int cat_cbytes, nbytes;
    cat_cbytes = nbytes = 0;

    hsize_t start[8],
            stride[8],
            count[8],
            block[8];

    hid_t           file_cat_w, file_cat_r, file_h5_w, file_h5_r, space, mem_space,
                    dset_cat_w, dset_cat_r, dset_h5_w, dset_h5_r, dcpl;    /* Handles */
    herr_t          status;
    unsigned        flt_msk = 0;

    // Create HDF5 datasets
    hid_t type_h5;
    switch (arr->itemsize) {
        case 4:
            type_h5 = H5T_STD_I32LE;
            break;
        case 8:
            type_h5 = H5T_STD_I64LE;
    }
    file_cat_w = H5Fcreate (FILE_CAT, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    file_h5_w = H5Fcreate (FILE_H5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    space = H5Screate_simple (ndim, (const hsize_t *) extshape, NULL);
    hsize_t memsize = (hsize_t) chunknelems;
    mem_space = H5Screate_simple (1, &memsize, NULL);
    dcpl = H5Pcreate (H5P_DATASET_CREATE);
    status = H5Pset_chunk (dcpl, ndim, chunks);
    dset_cat_w = H5Dcreate (file_cat_w, DATASET_CAT, type_h5, space, H5P_DEFAULT, dcpl,
                          H5P_DEFAULT);
    status = H5Pset_shuffle (dcpl);
    status = H5Pset_deflate (dcpl, 1);
    dset_h5_w = H5Dcreate (file_h5_w, DATASET_H5, type_h5, space, H5P_DEFAULT, dcpl,
                         H5P_DEFAULT);
    start[0] = 0;
    stride[0] = chunknelems;
    count[0] = 1;
    block[0] = chunknelems;
    status = H5Sselect_hyperslab (mem_space, H5S_SELECT_SET, start, stride, count, block);

    for(int nchunk = 0; nchunk < arr->sc->nchunks; nchunk++) {
        // Get chunk
        decompressed = blosc2_schunk_decompress_chunk(arr->sc, nchunk, chunk, (int32_t) chunksize);
        if (decompressed < 0) {
            printf("Error reading chunk \n");
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        } else {
            nbytes += decompressed;
        }

        // Compress chunk using Blosc + ZLIB + SHUFFLE
        blosc_set_timestamp(&t0);
        compressed = blosc2_compress_ctx(cctx, chunk, decompressed, cchunk, chunksize);
        if (compressed < 0) {
            printf("Error Caterva compress \n");
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        } else {
            cat_cbytes += compressed;
        }
        blosc2_schunk_append_chunk(wschunk, cchunk, true);
        blosc_set_timestamp(&t1);
        cat_time_w += blosc_elapsed_secs(t0, t1);

        // Use H5Dwrite to compress and save buffer using gzip
        blosc_set_timestamp(&t0);
        for (int i = 0; i < ndim; ++i) {
            start[i] = nchunk_ndim[i] * chunks[i];
            stride[i] = chunks[i];
            count[i] = 1;
            block[i] = chunks[i];
        }
        status = H5Sselect_hyperslab(space, H5S_SELECT_SET, start, stride, count, block);
        if (status < 0) {
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }
        status = H5Dwrite(dset_h5_w, H5T_NATIVE_INT, mem_space, space, H5P_DEFAULT,
                          chunk);
        blosc_set_timestamp(&t1);
        h5_time_w += blosc_elapsed_secs(t0, t1);
        if (status < 0) {
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }
    }

    // Convert Blosc superchunk to compressed frame
    blosc_set_timestamp(&t0);
    uint8_t *cframe = malloc(wschunk->nbytes);
    bool needs_free;
    int64_t frame_len = blosc2_schunk_to_buffer(wschunk, &cframe, &needs_free);

    // Use H5Dwrite_chunk to save Blosc compressed frame
    status = H5Dwrite_chunk(dset_cat_w, H5P_DEFAULT, flt_msk, offset, frame_len, cframe);
    if (status < 0) {
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }
    blosc_set_timestamp(&t1);
    cat_time_w += blosc_elapsed_secs(t0, t1);

    printf("nchunks: %ld", arr->sc->nchunks);
    printf("Caterva write: %f s\n", cat_time_w);
    printf("HDF5 write: %f s\n", h5_time_w);

    // Close and release resources.
    status = H5Pclose (dcpl);
    status = H5Sclose (space);
    status = H5Sclose (mem_space);
    status = H5Fclose (file_cat_w);
    status = H5Fclose (file_h5_w);
    status = H5Dclose (dset_cat_w);
    status = H5Dclose (dset_h5_w);

    // Open HDF5 datasets
    file_cat_r = H5Fopen (FILE_CAT, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_cat_r = H5Dopen (file_cat_r, DATASET_CAT, H5P_DEFAULT);
    file_h5_r = H5Fopen (FILE_H5, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_h5_r = H5Dopen (file_h5_r, DATASET_H5, H5P_DEFAULT);
    dcpl = H5Dget_create_plist (dset_h5_r);
    space = H5Screate_simple (ndim, (const hsize_t *) extshape, NULL);
    mem_space = H5Screate_simple (1, &memsize, NULL);
    start[0] = 0;
    stride[0] = chunknelems;
    count[0] = 1;
    block[0] = chunknelems;
    status = H5Sselect_hyperslab (mem_space, H5S_SELECT_SET, start, stride, count, block);
    int64_t cbufsize;

    // Read Blosc compressed frame
    blosc_set_timestamp(&t0);
    status = H5Dread_chunk(dset_cat_r, H5P_DEFAULT, offset, &flt_msk, cbuffer);
    if (status < 0) {
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }

    // Get frame len
    uint8_t aux_[8];
    for (int i = 0; i < 8; ++i) {
        aux_[i] = cbuffer[23 - i];
    }
    memcpy(&cbufsize, aux_, 8);

    // Get compressed superchunk
    rschunk = blosc2_schunk_from_buffer((uint8_t *) cbuffer, (int64_t) cbufsize, false);
    if (rschunk == NULL) {
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }
    blosc_set_timestamp(&t1);
    cat_time_r += blosc_elapsed_secs(t0, t1);

    for(int nchunk = 0; nchunk < arr->sc->nchunks; nchunk++) {
        // Read HDF5 buffer
        blosc_set_timestamp(&t0);
        for (int i = 0; i < ndim; ++i) {
            start[i] = nchunk_ndim[i] * chunks[i];
            stride[i] = chunks[i];
            count[i] = 1;
            block[i] = chunks[i];
        }
        status = H5Sselect_hyperslab (space, H5S_SELECT_SET, start, stride, count,
                                      block);
        if (status < 0) {
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }
        status = H5Dread (dset_h5_r, H5T_NATIVE_INT, mem_space, space, H5P_DEFAULT,
                          buffer_h5);
        blosc_set_timestamp(&t1);
        h5_time_r += blosc_elapsed_secs(t0, t1);
        if (status < 0) {
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }

        // Decompress chunk using Blosc + ZLIB + SHUFFLE
        blosc_set_timestamp(&t0);
        decompressed = blosc2_schunk_decompress_chunk(rschunk, nchunk, buffer_cat, chunksize);
        blosc_set_timestamp(&t1);
        cat_time_r += blosc_elapsed_secs(t0, t1);

        if (decompressed < 0) {
            printf("Error Caterva decompress \n");
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }

        // Check that every buffer is equal
        for (int k = 0; k < decompressed / arr->itemsize; ++k) {
            if (buffer_h5[k] != buffer_cat[k]) {
                printf("Input not equal to output: %d, %d \n", buffer_cat[k], buffer_h5[k]);
            }
        }
    }

    printf("Caterva read: %f s\n", cat_time_r);
    printf("HDF5 read: %f s\n", h5_time_r);

    // Close and release resources.
    status = H5Sclose (space);
    status = H5Sclose (mem_space);
    status = H5Pclose (dcpl);
    status = H5Dclose (dset_cat_r);
    status = H5Fclose (file_cat_r);
    status = H5Dclose (dset_h5_r);
    status = H5Fclose (file_h5_r);
    free(chunk);
    free(cchunk);
    free(buffer_cat);
    free(cbuffer);
    free(buffer_h5);
    caterva_free(ctx, &arr);
    caterva_ctx_free(&ctx);
    blosc2_schunk_free(wschunk);
    blosc2_schunk_free(rschunk);
    if (needs_free) {
        free(cframe);
    }

    blosc_destroy();

    return 0;
}


int solar1() {
    int result = comp("../../bench/solar1.cat", "solar1_cat.b2frame");
    return result;
}

int air1() {
    int result = comp("../../bench/air1.cat", "air1_cat.b2frame");
    return result;
}

int snow1() {
    int result = comp("../../bench/snow1.cat", "snow1_cat.b2frame");
    return result;
}

int wind1() {
    int result = comp("../../bench/wind1.cat", "wind1_cat.b2frame");
    return result;
}

int precip1() {
    int result = comp("../../bench/precip1.cat", "precip1_cat.b2frame");
    return result;
}

int precip2() {
    int result = comp("../../bench/precip2.cat", "precip2_cat.b2frame");
    return result;
}

int precip3() {
    int result = comp("../../bench/precip3.cat", "precip3_cat.b2frame");
    return result;
}

int precip3m() {
    int result = comp("../../bench/precip-3m.cat", "precip3m_cat.b2frame");
    return result;
}

int easy() {
    int result = comp("../../bench/easy.caterva", "easy_cat.b2frame");
    return result;
}

int cyclic() {
    int result = comp("../../bench/cyclic.caterva", "cyclic_cat.b2frame");
    return result;
}

int main() {

    unsigned majnum, minnum, vers;
    if (H5get_libversion(&majnum, &minnum, &vers) >= 0)
        printf("VERSION %d.%d.%d \n", majnum, minnum, vers);

    printf("cyclic \n");
    CATERVA_ERROR(cyclic());
 /*   printf("easy \n");
    CATERVA_ERROR(easy());
    printf("wind1 \n");
    CATERVA_ERROR(wind1());
  *  printf("air1 \n");
    CATERVA_ERROR(air1());
   * printf("solar1 \n");
    CATERVA_ERROR(solar1());
  *  printf("snow1 \n");
    CATERVA_ERROR(snow1());
  *  printf("precip1 \n");
    CATERVA_ERROR(precip1());
    /*  printf("precip2 \n");
    CATERVA_ERROR(precip2());
    printf("precip3 \n");
    CATERVA_ERROR(precip3());
    printf("precip3m \n");
    CATERVA_ERROR(precip3m());
    return CATERVA_SUCCEED;
*/
}
