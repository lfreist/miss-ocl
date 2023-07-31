add_library(opencl opencl.cpp)
target_link_libraries(opencl PUBLIC OpenCL::OpenCL)