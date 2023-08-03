add_executable(OpenCL_Devices devices.cpp)
target_link_libraries(OpenCL_Devices PRIVATE opencl)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE opencl)