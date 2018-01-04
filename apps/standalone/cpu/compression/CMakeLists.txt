find_package(ZFP)
if (ZFP_FOUND)
   add_definitions(-DGADGETRON_COMPRESSION_ZFP)
endif()

#add_executable(compression standalone_compression.cpp)

#add_executable(ismrmrd_compression ismrmrd_compression.cpp)

#target_link_libraries(compression gadgetron_toolbox_mri_core gadgetron_toolbox_hostutils gadgetron_toolbox_cpucore ${ISMRMRD_LIBRARIES})
if (ZFP_FOUND)
   #target_link_libraries(compression ${ZFP_LIBRARIES})
endif ()

#target_link_libraries(ismrmrd_compression gadgetron_toolbox_mri_core gadgetron_toolbox_hostutils gadgetron_toolbox_cpucore ${ISMRMRD_LIBRARIES})

#install(TARGETS compression ismrmrd_compression DESTINATION bin COMPONENT main)

