cmake_minimum_required(VERSION 3.15)
project(LouvainBenchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the Boost.Graph fork with louvain_clustering.hpp
# Override at configure time:  cmake .. -DBGL_GRAPH_INCLUDE=<path>
set(BGL_GRAPH_INCLUDE "${CMAKE_SOURCE_DIR}/../../../forks/graph/include"
    CACHE PATH "Path to Boost.Graph fork include/ directory")

# Find Boost (system install for core headers)
find_package(Boost REQUIRED)

# Include directories — fork first so its headers take precedence
include_directories(${BGL_GRAPH_INCLUDE})
include_directories(${Boost_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# All BGL graph-type variants share bgl_louvain_benchmark.hpp.
# Each .cpp is a thin wrapper that instantiates the template for one variant.
# ---------------------------------------------------------------------------

set(BGL_VARIANTS
    bgl_louvain_vecS_vecS     # vecS/vecS
    bgl_louvain_listS_vecS    # listS/vecS
    bgl_louvain_setS_vecS     # setS/vecS
    bgl_louvain_matrix         # adjacency_matrix
)

# Non-incremental quality function variants (force full modularity recomputation)
set(BGL_VARIANTS_NOINC
    bgl_louvain_vecS_vecS_noinc
    bgl_louvain_listS_vecS_noinc
    bgl_louvain_setS_vecS_noinc
    bgl_louvain_matrix_noinc
)

# "Trust aggregated Q" variants — compile the same source files but with
# BOOST_GRAPH_LOUVAIN_TRUST_AGGREGATED_Q=1.  This skips the per-level
# recomputation of Q on the original graph and the best-partition tracking.
set(BGL_TRUST_TARGETS
    bgl_louvain_vecS_vecS_trust
    bgl_louvain_listS_vecS_trust
    bgl_louvain_setS_vecS_trust
    bgl_louvain_matrix_trust
)
set(BGL_TRUST_SOURCES
    bgl_louvain_vecS_vecS.cpp
    bgl_louvain_listS_vecS.cpp
    bgl_louvain_setS_vecS.cpp
    bgl_louvain_matrix.cpp
)

# "Track peak Q" variants — TRACK_PEAK_Q=1 only.
# Recomputes Q on the original graph (safe) and keeps the best partition.
set(BGL_PEAK_TARGETS
    bgl_louvain_vecS_vecS_peak
    bgl_louvain_listS_vecS_peak
    bgl_louvain_setS_vecS_peak
    bgl_louvain_matrix_peak
)
set(BGL_PEAK_SOURCES
    bgl_louvain_vecS_vecS.cpp
    bgl_louvain_listS_vecS.cpp
    bgl_louvain_setS_vecS.cpp
    bgl_louvain_matrix.cpp
)

# "Trust + Peak" variants — both TRUST_AGGREGATED_Q=1 and TRACK_PEAK_Q=1.
# Trusts aggregated Q (no recomputation) but still tracks the best partition.
set(BGL_TRUST_PEAK_TARGETS
    bgl_louvain_vecS_vecS_trust_peak
    bgl_louvain_listS_vecS_trust_peak
    bgl_louvain_setS_vecS_trust_peak
    bgl_louvain_matrix_trust_peak
)
set(BGL_TRUST_PEAK_SOURCES
    bgl_louvain_vecS_vecS.cpp
    bgl_louvain_listS_vecS.cpp
    bgl_louvain_setS_vecS.cpp
    bgl_louvain_matrix.cpp
)

# NOTE: listS-as-VertexList variants (vecS_listS, listS_listS) are not
# included here because they are uncommon in practice. The refactored
# louvain_clustering.hpp now uses vertex_index maps internally and
# supports arbitrary VertexList selectors.

foreach(variant ${BGL_VARIANTS})
    add_executable(${variant} ${variant}.cpp)
    target_link_libraries(${variant} ${Boost_LIBRARIES})
    target_compile_options(${variant} PRIVATE -O3)
endforeach()

foreach(variant ${BGL_VARIANTS_NOINC})
    add_executable(${variant} ${variant}.cpp)
    target_link_libraries(${variant} ${Boost_LIBRARIES})
    target_compile_options(${variant} PRIVATE -O3)
endforeach()

list(LENGTH BGL_TRUST_TARGETS _trust_count)
math(EXPR _trust_last "${_trust_count} - 1")
foreach(i RANGE 0 ${_trust_last})
    list(GET BGL_TRUST_TARGETS ${i} target_name)
    list(GET BGL_TRUST_SOURCES ${i} source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} ${Boost_LIBRARIES})
    target_compile_options(${target_name} PRIVATE -O3)
    target_compile_definitions(${target_name} PRIVATE BOOST_GRAPH_LOUVAIN_TRUST_AGGREGATED_Q=1)
endforeach()

list(LENGTH BGL_PEAK_TARGETS _peak_count)
math(EXPR _peak_last "${_peak_count} - 1")
foreach(i RANGE 0 ${_peak_last})
    list(GET BGL_PEAK_TARGETS ${i} target_name)
    list(GET BGL_PEAK_SOURCES ${i} source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} ${Boost_LIBRARIES})
    target_compile_options(${target_name} PRIVATE -O3)
    target_compile_definitions(${target_name} PRIVATE BOOST_GRAPH_LOUVAIN_TRACK_PEAK_Q=1)
endforeach()

list(LENGTH BGL_TRUST_PEAK_TARGETS _tp_count)
math(EXPR _tp_last "${_tp_count} - 1")
foreach(i RANGE 0 ${_tp_last})
    list(GET BGL_TRUST_PEAK_TARGETS ${i} target_name)
    list(GET BGL_TRUST_PEAK_SOURCES ${i} source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} ${Boost_LIBRARIES})
    target_compile_options(${target_name} PRIVATE -O3)
    target_compile_definitions(${target_name} PRIVATE
        BOOST_GRAPH_LOUVAIN_TRUST_AGGREGATED_Q=1
        BOOST_GRAPH_LOUVAIN_TRACK_PEAK_Q=1)
endforeach()

# ---------------------------------------------------------------------------
# Gen-louvain reference implementation (built from vendored / downloaded source)
# ---------------------------------------------------------------------------

set(GENLOUVAIN_SRC_DIR "${CMAKE_SOURCE_DIR}/../vendor/gen-louvain/gen-louvain/src"
    CACHE PATH "Path to gen-louvain extracted source directory")

if(EXISTS "${GENLOUVAIN_SRC_DIR}/main_louvain.cpp")
    message(STATUS "Found gen-louvain source in: ${GENLOUVAIN_SRC_DIR}")

    # Shared object library for the core louvain engine
    set(GENLOUVAIN_CORE_SRCS
        ${GENLOUVAIN_SRC_DIR}/graph_binary.cpp
        ${GENLOUVAIN_SRC_DIR}/louvain.cpp
        ${GENLOUVAIN_SRC_DIR}/quality.cpp
        ${GENLOUVAIN_SRC_DIR}/modularity.cpp
        ${GENLOUVAIN_SRC_DIR}/zahn.cpp
        ${GENLOUVAIN_SRC_DIR}/owzad.cpp
        ${GENLOUVAIN_SRC_DIR}/goldberg.cpp
        ${GENLOUVAIN_SRC_DIR}/condora.cpp
        ${GENLOUVAIN_SRC_DIR}/devind.cpp
        ${GENLOUVAIN_SRC_DIR}/devuni.cpp
        ${GENLOUVAIN_SRC_DIR}/dp.cpp
        ${GENLOUVAIN_SRC_DIR}/shimalik.cpp
        ${GENLOUVAIN_SRC_DIR}/balmod.cpp
    )

    add_library(genlouvain_core STATIC ${GENLOUVAIN_CORE_SRCS})
    target_include_directories(genlouvain_core PUBLIC ${GENLOUVAIN_SRC_DIR})
    target_compile_options(genlouvain_core PRIVATE -O3 -w)

    # gen-louvain executables
    add_executable(genlouvain_louvain ${GENLOUVAIN_SRC_DIR}/main_louvain.cpp)
    target_link_libraries(genlouvain_louvain genlouvain_core)
    target_compile_options(genlouvain_louvain PRIVATE -O3 -w)

    add_executable(genlouvain_convert ${GENLOUVAIN_SRC_DIR}/main_convert.cpp
                                     ${GENLOUVAIN_SRC_DIR}/graph.cpp)
    target_include_directories(genlouvain_convert PRIVATE ${GENLOUVAIN_SRC_DIR})
    target_compile_options(genlouvain_convert PRIVATE -O3 -w)

    add_executable(genlouvain_hierarchy ${GENLOUVAIN_SRC_DIR}/main_hierarchy.cpp)
    target_include_directories(genlouvain_hierarchy PRIVATE ${GENLOUVAIN_SRC_DIR})
    target_compile_options(genlouvain_hierarchy PRIVATE -O3 -w)

else()
    message(STATUS "Gen-louvain source not found at ${GENLOUVAIN_SRC_DIR} — skipping.")
    message(STATUS "  Run: bash vendor/gen-louvain/build.sh  to download and patch the source.")
endif()
