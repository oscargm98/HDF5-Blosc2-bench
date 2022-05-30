cmake_minimum_required (VERSION 3.12)
project (HDF5-Blosc2_C_H5D)

file(GLOB SOURCES h5ex_d_*.c)

foreach(source ${SOURCES})
  get_filename_component(target_name ${source} NAME_WE)
  set(target ${target_name})
  add_executable(${target} ${target_name}.c)
  target_link_libraries(${target} HDF5_LIBRARY)
  target_link_libraries(${target} CAT_LIBRARY)
  target_link_libraries(${target} BLOSC2_LIBRARY)
endforeach(source)
