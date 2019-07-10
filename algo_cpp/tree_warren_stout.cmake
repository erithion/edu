cmake_minimum_required      ( VERSION 3.12.0 )

set                         ( PROJECT_NAME                       tree_warren_stout )
set                         ( INSTALLDIR                         ${BUILDDIR}/cpp/ )
set                         ( SRC_FILES                          tree_warren_stout.cpp)
set                         ( INC_FILES                          tree_binary.hpp tree_warren_stout.hpp)

# compiler/linker flags
string                      ( APPEND CMAKE_CXX_FLAGS             " /EHsc" )
string                      ( APPEND CMAKE_CXX_FLAGS             " /std:c++14" )
string                      ( APPEND CMAKE_SHARED_LINKER_FLAGS   " /SAFESEH:NO" )
message                     ( STATUS                             "CXX flags ${CMAKE_CXX_FLAGS}" )
message                     ( STATUS                             "Linker flags ${CMAKE_SHARED_LINKER_FLAGS}" )
         
project                     ( ${PROJECT_NAME} )

include_directories         ( ${CMAKE_CURRENT_SOURCE_DIR} )
     

add_executable              ( ${PROJECT_NAME}                    ${SRC_FILES} 
                                                                 ${INC_FILES} )

target_link_libraries       ( ${PROJECT_NAME}                     )

install                     ( TARGETS                            ${PROJECT_NAME}                         
                              DESTINATION                        ${INSTALLDIR} )
