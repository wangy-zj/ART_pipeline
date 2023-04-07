#add_executable(pipeline_1ant1pol pipeline_1ant1pol.cu ${SOURCE_FILES})
#target_include_directories(pipeline_1ant1pol PRIVATE $ENV{HOME}/.local/include)
#target_link_libraries(pipeline_1ant1pol ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
#set_target_properties(pipeline_1ant1pol PROPERTIES LINKER_LANGUAGE CUDA)

#add_executable(pipeline_dada_1ant1pol pipeline_dada_1ant1pol.cu ${SOURCE_FILES})
#target_include_directories(pipeline_dada_1ant1pol PRIVATE $ENV{HOME}/.local/include)
#target_link_libraries(pipeline_dada_1ant1pol ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
#set_target_properties(pipeline_dada_1ant1pol PROPERTIES LINKER_LANGUAGE CUDA)

add_executable(pipeline_dada_amplitude_phase pipeline_dada_amplitude_phase.cu ${SOURCE_FILES})
target_include_directories(pipeline_dada_amplitude_phase PRIVATE $ENV{HOME}/.local/include)
target_link_libraries(pipeline_dada_amplitude_phase ${CUDA_LIBRARIES} ${PSRDADA_LIBRARY} OpenMP::OpenMP_CXX udppipe)
set_target_properties(pipeline_dada_amplitude_phase PROPERTIES LINKER_LANGUAGE CUDA)