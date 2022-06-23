# HDF5-Blosc2

To run the benchmarks with vmtouch, you just have to uncomment the next block of code:

    char command[50];
    strcpy(command, "vmtouch -e h5ex_cat.h5 h5ex_h5.h5" );
    system(command);

It is located between the closing process of HDF5 writing resources and the opening process of HDF5 reading resources.