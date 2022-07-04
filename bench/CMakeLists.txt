cmake_minimum_required (VERSION 3.18)
project (HDF5-Blosc2_C_H5D)

file(GLOB SOURCES chunk_gzip.c gzip_vs_cat*.c)

include_directories("/home/soscar/miniconda3/include")


foreach(source ${SOURCES})
  get_filename_component(target_name ${source} NAME_WE)
  set(target ${target_name})
  add_executable(${target} ${target_name}.c)
  target_link_libraries(${target} HDF5_LIBRARY)
  target_link_libraries(${target} CAT_LIBRARY)
  target_link_libraries(${target} BLOSC2_LIBRARY)
endforeach(source)
