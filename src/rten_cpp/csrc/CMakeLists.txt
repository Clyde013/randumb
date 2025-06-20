cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

list(APPEND CMAKE_PREFIX_PATH "~/anaconda3/envs/randumb/lib/python3.11/site-packages/torch/share/cmake")
# https://stackoverflow.com/a/46515110
set(DCUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.5/")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.5/bin/nvcc")

project(example-app)

# Also, enable CUDA language support.
# enable_language(CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# https://stackoverflow.com/questions/67440802/point-cmake-to-proper-python-header-inside-conda-env
# https://github.com/microsoft/vscode-cmake-tools/issues/1915#issuecomment-1141687971
# like what the fuck vscode. if vscode is bugging tf out, just rerun the build command on cmdline with the conda env
# active and it should just fix stuff.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})


add_executable(main main.cpp cuda/rten.cuh cuda/rten.cu)
set_property(TARGET main PROPERTY CXX_STANDARD 17)

# link external torch libs
target_link_libraries(main "${TORCH_LIBRARIES}" Python3::Python)

# include cuda/ directory with all the cuda kernel code and .cuh
target_include_directories(main PRIVATE cuda)

# nvcc compilation flags
# for -lineinfo to debug with ncu, ensure that the cmake build is set to "release"
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET main
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:main>)
endif (MSVC)