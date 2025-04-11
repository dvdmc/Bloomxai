
add_library(bonxai_core INTERFACE
    include/bonxai/bonxai.hpp
    include/bonxai/serialization.hpp
    include/bonxai/grid_coord.hpp
)
set_target_properties(bonxai_core PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(bonxai_core PUBLIC INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

    
add_library(bloomxai_map
    include/bloomxai_map/pcl_utils.hpp
    include/bloomxai_map/semantic_map.hpp
    src/pcl_utils.cpp
    src/semantic_map.cpp)

target_compile_definitions(bloomxai_map PRIVATE
    ${PCL_DEFINITIONS})

target_include_directories(bloomxai_map PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_link_libraries(bloomxai_map PUBLIC
    bonxai_core
    ${PCL_LIBRARIES})

####################################################

if(benchmark_FOUND AND octomap_FOUND)
    add_subdirectory(benchmark)
endif()

