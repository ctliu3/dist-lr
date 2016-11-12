aux_source_directory(. DIR_LIB_SRCS)

set(LIBS ${pslite_LINKER_LIBS} pslite zmq)
add_executable(distlr ${DIR_LIB_SRCS})
target_link_libraries(distlr ${LIBS})
