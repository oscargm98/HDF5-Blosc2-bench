/************************************************************

  This benchmark compares times gotten during reading and
  writing data to a dataset process using:
  - HDF5 with gzip + shuffle compression.
  - Blosc2 with zlib + shuffle compression
  processing the whole superchunk at once.

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

int comp(char* urlpath_input)
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

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode = BLOSC_ZLIB;
    cparams.typesize = arr->itemsize;
    cparams.clevel = 1;
    cparams.nthreads = 6;
    cparams.blocksize = arr->sc->blocksize;
    blosc2_context *cctx;
    cctx = blosc2_create_cctx(cparams);
    blosc2_storage storage = {.cparams=&cparams, .contiguous=false, .urlpath = NULL};
    blosc2_schunk* wschunk = blosc2_schunk_new(&storage);

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
    if ((void *) dset_cat_w == NULL) {
        printf("Can not create HDF5 stream \n");
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }
    status = H5Pset_shuffle (dcpl);
    status = H5Pset_deflate (dcpl, 1);
    dset_h5_w = H5Dcreate (file_h5_w, DATASET_H5, type_h5, space, H5P_DEFAULT, dcpl,
                         H5P_DEFAULT);
    if ((void *) dset_h5_w == NULL) {
        printf("Can not create HDF5 stream \n");
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }
    start[0] = 0;
    stride[0] = chunknelems;
    count[0] = 1;
    block[0] = chunknelems;
    status = H5Sselect_hyperslab (mem_space, H5S_SELECT_SET, start, stride, count, block);
    if (status < 0) {
        return -1;
    }

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
        int64_t append = blosc2_schunk_append_chunk(wschunk, cchunk, true);
        if (append < 0) {
            printf("Error appending chunk \n");
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }
        blosc_set_timestamp(&t1);
        cat_time_w += blosc_elapsed_secs(t0, t1);

        // Use H5Dwrite to compress and save buffer using gzip
        blosc_set_timestamp(&t0);
        blosc2_unidim_to_multidim((int8_t) ndim, (int64_t *) chunksdim, nchunk, (int64_t *) nchunk_ndim);
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

    printf("nchunks: %ld\n", arr->sc->nchunks);
    printf("Caterva write: %f s\n", cat_time_w);
    printf("HDF5 write: %f s\n", h5_time_w);

    // Close and release resources.
    H5Pclose (dcpl);
    H5Sclose (space);
    H5Sclose (mem_space);
    H5Fclose (file_cat_w);
    H5Fclose (file_h5_w);
    H5Dclose (dset_cat_w);
    H5Dclose (dset_h5_w);
/*
    char command[50];
    strcpy(command, "vmtouch -e h5ex_cat.h5 h5ex_h5.h5" );
    system(command);
*/
    // Open HDF5 datasets
    file_cat_r = H5Fopen (FILE_CAT, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_cat_r = H5Dopen (file_cat_r, DATASET_CAT, H5P_DEFAULT);
    if ((void *) dset_cat_r == NULL) {
        printf("Can not open HDF5 stream \n");
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }
    file_h5_r = H5Fopen (FILE_H5, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_h5_r = H5Dopen (file_h5_r, DATASET_H5, H5P_DEFAULT);
    if ((void *) dset_h5_r == NULL) {
        printf("Can not open HDF5 stream \n");
        free(chunk);
        free(cchunk);
        free(buffer_cat);
        free(cbuffer);
        free(buffer_h5);
        return -1;
    }
    dcpl = H5Dget_create_plist (dset_h5_r);
    space = H5Screate_simple (ndim, (const hsize_t *) extshape, NULL);
    mem_space = H5Screate_simple (1, &memsize, NULL);
    start[0] = 0;
    stride[0] = chunknelems;
    count[0] = 1;
    block[0] = chunknelems;
    status = H5Sselect_hyperslab (mem_space, H5S_SELECT_SET, start, stride, count, block);

    // Get compressed superchunk
    blosc_set_timestamp(&t0);
    haddr_t address = 0;
    hid_t chunk_space = H5Dget_space(dset_cat_r);
    hsize_t cframe_size;
    H5Dget_chunk_info(dset_cat_r, chunk_space, 0, NULL, &flt_msk, &address, &cframe_size);
    blosc2_schunk* rschunk = blosc2_schunk_open_offset(FILE_CAT, (int64_t) address);
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
    blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS;
    dparams.nthreads = 6;
    dparams.schunk = rschunk;
    blosc2_context *dctx;
    dctx = blosc2_create_dctx(dparams);
    bool needs_free2;

    for(int nchunk = 0; nchunk < arr->sc->nchunks; nchunk++) {
        // Read HDF5 buffer
        blosc_set_timestamp(&t0);
        blosc2_unidim_to_multidim((int8_t) ndim, (int64_t *) chunksdim, nchunk, (int64_t *) nchunk_ndim);
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
  //      decompressed = blosc2_schunk_decompress_chunk(rschunk, nchunk, buffer_cat, chunksize);
        compressed = blosc2_schunk_get_lazychunk(rschunk, nchunk, &cchunk, &needs_free2);
        if (compressed < 0) {
            printf("Can not get lazy chunk %d \n", nchunk);
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }
        decompressed = blosc2_decompress_ctx(dctx, cchunk, compressed, buffer_cat, chunksize);
        if (decompressed < 0) {
            printf("Can not decompress chunk %d \n", nchunk);
            free(chunk);
            free(cchunk);
            free(buffer_cat);
            free(cbuffer);
            free(buffer_h5);
            return -1;
        }
        blosc_set_timestamp(&t1);
        cat_time_r += blosc_elapsed_secs(t0, t1);

        // Check that both buffers are equal
        for (int k = 0; k < decompressed / arr->itemsize; ++k) {
            if (buffer_h5[k] != buffer_cat[k]) {
                printf("HDF5 output not equal to Blosc output: %d, %d \n", buffer_h5[k], buffer_cat[k]);
                return -1;
            }
        }
    }

    printf("Caterva read: %f s\n", cat_time_r);
    printf("HDF5 read: %f s\n", h5_time_r);

    // Close and release resources.
    H5Sclose (space);
    H5Sclose (mem_space);
    H5Pclose (dcpl);
    H5Dclose (dset_cat_r);
    H5Fclose (file_cat_r);
    H5Dclose (dset_h5_r);
    H5Fclose (file_h5_r);
    free(chunk);
    if (needs_free2){
        free(cchunk);
    }
    free(buffer_cat);
    free(cbuffer);
    free(buffer_h5);
    blosc2_free_ctx(dctx);
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
    int result = comp("../../data/solar1.cat");
    return result;
}

int air1() {
    int result = comp("../../data/air1.cat");
    return result;
}

int snow1() {
    int result = comp("../../data/snow1.cat");
    return result;
}

int wind1() {
    int result = comp("../../data/wind1.cat");
    return result;
}

int precip1() {
    int result = comp("../../data/precip1.cat");
    return result;
}

int precip2() {
    int result = comp("../../data/precip2.cat");
    return result;
}

int precip3() {
    int result = comp("../../data/precip3.cat");
    return result;
}

int precip3m() {
    int result = comp("../../data/precip-3m.cat");
    return result;
}

int easy() {
    int result = comp("../../data/easy.cat");
    return result;
}

int cyclic() {
    int result = comp("../../data/cyclic.cat");
    return result;
}

int main() {

    unsigned majnum, minnum, vers;
    if (H5get_libversion(&majnum, &minnum, &vers) >= 0)
        printf("HDF5 working with version %d.%d.%d \n", majnum, minnum, vers);
/*
    printf("cyclic \n");
    CATERVA_ERROR(cyclic());
 */   printf("easy \n");
    CATERVA_ERROR(easy());
    printf("wind1 \n");
    CATERVA_ERROR(wind1());
    printf("air1 \n");
    CATERVA_ERROR(air1());
    printf("solar1 \n");
    CATERVA_ERROR(solar1());
    printf("snow1 \n");
    CATERVA_ERROR(snow1());
    printf("precip1 \n");
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
