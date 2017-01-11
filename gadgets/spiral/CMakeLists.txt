if (WIN32)
  add_definitions(-D__BUILD_GADGETRON_SPIRAL__)
endif ()

include_directories(
  ${CMAKE_SOURCE_DIR}/gadgets/mri_core
  ${CMAKE_SOURCE_DIR}/gadgets/pmri
  ${CMAKE_SOURCE_DIR}/toolboxes/nfft/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/fft/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/cpu
  ${CMAKE_SOURCE_DIR}/toolboxes/core/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/mri_core
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers
  ${CMAKE_SOURCE_DIR}/toolboxes/solvers/gpu
  ${CMAKE_SOURCE_DIR}/toolboxes/operators
  ${CMAKE_SOURCE_DIR}/toolboxes/operators/gpu
  ${ARMADILLO_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  )

add_library(gadgetron_spiral SHARED 
  gadgetron_spiral_export.h 
  vds.cpp 
  gpuSpiralSensePrepGadget.h gpuSpiralSensePrepGadget.cpp 
  SpiralToGenericGadget.h SpiralToGenericGadget.cpp
  gpuSpiralDeblurGadget.h gpuSpiralDeblurGadget.cpp
  gpuSpiralAutoDeblurGadget.h gpuSpiralAutoDeblurGadget.cpp
  gpuSpiralDeblurGadgetClean.h gpuSpiralDeblurGadgetClean.cpp
  gpuSpiralDeblurGadgetv2.h gpuSpiralDeblurGadgetv2.cpp)

set_target_properties(gadgetron_spiral PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(gadgetron_spiral
  gadgetron_gadgetbase
  gadgetron_toolbox_log
  gadgetron_toolbox_cpucore gadgetron_toolbox_gpucore gadgetron_toolbox_mri_core gadgetron_toolbox_gpunfft gadgetron_toolbox_gpusolvers gadgetron_toolbox_gpuoperators gadgetron_toolbox_cpufft
  ${ISMRMRD_LIBRARIES} ${FFTW3_LIBRARIES} ${CUDA_LIBRARIES} ${ARMADILLO_LIBRARIES}
  optimized ${ACE_LIBRARIES} debug ${ACE_DEBUG_LIBRARY} 
  )

target_link_libraries(gadgetron_spiral gadgetron_toolbox_gpuparallelmri)

install (TARGETS gadgetron_spiral DESTINATION lib COMPONENT main)
install (FILES vds.h gadgetron_spiral_export.h 
					 gpuSpiralDeblurGadget.h
					 gpuSpiralDeblurGadgetClean.h
					 gpuSpiralDeblurGadgetv2.h
					 gpuSpiralAutoDeblurGadget.h
                     gpuSpiralSensePrepGadget.h 
                     SpiralToGenericGadget.h 
                     DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)

add_subdirectory(config)
