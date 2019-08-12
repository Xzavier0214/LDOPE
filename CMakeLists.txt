cmake_minimum_required(VERSION 3.0.0)
project(ldope VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(NLopt)
find_package(GSL)

include_directories(${PROJECT_SOURCE_DIR}/inc)
aux_source_directory(${PROJECT_SOURCE_DIR}/src LDOPE_SCRS)

add_library(ldope SHARED ${LDOPE_SCRS})
add_executable(test test.cpp ${LDOPE_SCRS})

target_link_libraries(ldope ${GSL_LIBRARIES} ${NLOPT_LIBRARIES})
target_link_libraries(test ${GSL_LIBRARIES} ${NLOPT_LIBRARIES})
