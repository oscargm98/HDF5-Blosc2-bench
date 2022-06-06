cmake_minimum_required(VERSION 3.18)
project(HDF5-Blosc2)

include_directories("/mnt/c/Users/sosca/CLionProjects/build/include")

add_library(HDF5_LIBRARY SHARED IMPORTED)
add_library(CAT_LIBRARY SHARED IMPORTED)
add_library(BLOSC2_LIBRARY SHARED IMPORTED)
set_target_properties(HDF5_LIBRARY PROPERTIES IMPORTED_LOCATION "/home/soscar/miniconda3/lib/libhdf5.so")
set_target_properties(CAT_LIBRARY PROPERTIES IMPORTED_LOCATION "/mnt/c/Users/sosca/CLionProjects/build/libs/debug/libcaterva.so")
set_target_properties(BLOSC2_LIBRARY PROPERTIES IMPORTED_LOCATION "/mnt/c/Users/sosca/CLionProjects/build/libs/debug/libblosc2.so")


add_subdirectory(bench)
add_subdirectory(H5D)