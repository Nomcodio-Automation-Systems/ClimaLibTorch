
cmake_minimum_required (VERSION 3.8)

project ("ClimaLibTorch")

#Include to libtorch path change to your path
set(CMAKE_PREFIX_PATH "../libtorch-debug2.5.1cu124/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable (ClimaLibTorch "ClimaLibTorch.cpp" "ClimaLibTorch.h" "ClimaNet.h" "ClimateDataset.h" "ClimateDataset.cpp" "CSVLoadingException.h" "Options.h" "RTools.h" "RTools.cpp" )
target_link_libraries(ClimaLibTorch "${TORCH_LIBRARIES}")
set_property(TARGET ClimaLibTorch PROPERTY CXX_STANDARD 17)
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()



if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ClimaLibTorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:ClimaLibTorch>)
endif (MSVC)
