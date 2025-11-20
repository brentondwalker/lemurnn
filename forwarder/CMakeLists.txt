#
# mkdir build
# cd build/
# cmake -DCMAKE_PREFIX_PATH=./libtorch ..
# make
#
cmake_minimum_required(VERSION 3.16)
project(lemu_forwarder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# package dependencies
find_package(PkgConfig REQUIRED)
find_package(Torch REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# executable target
add_executable(lemu_forwarder lemu_forwarder.cpp lemurnn.cpp)

# DPDK include directories
target_include_directories(lemu_forwarder PRIVATE ${DPDK_INCLUDE_DIRS})

# DPDK library directories
target_link_directories(lemu_forwarder PRIVATE ${DPDK_LIBRARY_DIRS})

# DPDK  and torch libraries to link against
# for some reason i get an error if I use two separate lines for this
target_link_libraries(lemu_forwarder PRIVATE ${DPDK_LIBRARIES} ${TORCH_LIBRARIES})

target_compile_options(lemu_forwarder PRIVATE ${DPDK_CFLAGS_OTHER})

target_link_options(lemu_forwarder PRIVATE ${DPDK_LDFLAGS_OTHER})

# Optional: Print a message upon successful configuration
message(STATUS "Configured lemu_forwarder")
message(STATUS "  DPDK Include Dirs: ${DPDK_INCLUDE_DIRS}")
message(STATUS "  DPDK Libraries: ${DPDK_LIBRARIES}")
