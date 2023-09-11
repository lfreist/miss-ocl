add_executable(OpenCL_Devices devices.cpp)
target_link_libraries(OpenCL_Devices PUBLIC miss-opencl)

add_executable(example example.cpp)
target_link_libraries(example PUBLIC miss-opencl)
