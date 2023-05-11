include (FindPackageHandleStandardArgs)

find_package(Threads REQUIRED)
find_package(MPI MODULE REQUIRED)

set(TAU_ROOT "${TAU_ROOT}"
  CACHE PATH "TAU root directory")
set(TAU_BUILD "${TAU_ROOT}/x86_64"
  CACHE PATH "TAU build directory")
set(OMP_PATH "${TAU_BUILD}/lib/shared-phase-ompt-mpi-pdt-openmp"
  CACHE PATH "TAU OpenMP directory")
set(BFD_ROOT "${TAU_BUILD}/binutils-2.36"
  CACHE PATH "TAU BFD directory")
set(UNWIND_ROOT "${TAU_BUILD}/libunwind-1.3.1-mpicc" 
  CACHE PATH "TAU libunwind directory")
set(OTF_ROOT "${TAU_BUILD}/otf2-mpicc"
  CACHE PATH "TAU OTF2 directory")
set(DWARF_ROOT "${TAU_BUILD}/libdwarf-mpicc"
  CACHE PATH "TAU DWARF directory")

set(TAU_LIBS
  TAU::TauMpi
  TAU::unwind
  TAU::OTF2
  TAU::DWARF
  TAU::elf
  dl m rt stdc++ z Threads::Threads
  )

set(TAU_COMPILE_DEFS 
  PROFILING_ON 
  TAU_GNU 
  TAU_DOT_H_LESS_HEADERS 
  TAU_MPI 
  TAU_UNIFY 
  TAU_MPI_THREADED 
  TAU_LINUX_TIMERS 
  TAU_MPIGREQUEST 
  TAU_MPIDATAREP 
  TAU_MPIERRHANDLER 
  TAU_MPICONSTCHAR 
  TAU_MPIATTRFUNCTION 
  TAU_MPITYPEEX 
  TAU_MPIADDERROR 
  TAU_LARGEFILE 
  _LARGEFILE64_SOURCE 
  TAU_BFD 
  TAU_MPIFILE 
  HAVE_GNU_DEMANGLE 
  HAVE_TR1_HASH_MAP 
  TAU_SS_ALLOC_SUPPORT 
  EBS_CLOCK_RES=1 
  TAU_STRSIGNAL_OK 
  TAU_UNWIND 
  TAU_USE_LIBUNWIND 
  TAU_TRACK_LD_LOADER 
  TAU_OPENMP_NESTED 
  TAU_USE_OMPT_5_0 
  TAU_USE_TLS 
  TAU_MPICH3 
  TAU_MPI_EXTENSIONS 
  TAU_OTF2 
  TAU_ELF_BFD 
  TAU_DWARF 
  TAU_OPENMP 
  TAU_UNIFY)

find_library(BFD_LIB
  NAMES bfd
  PATHS "${BFD_ROOT}" PATH_SUFFIXES lib
  REQUIRED)

find_path(IBERTY_INCLUDE_DIR
  NAMES libiberty.h
  PATHS "${BFD_ROOT}" PATH_SUFFIXES include
  REQUIRED)

find_library(IBERTY_LIB
  NAMES iberty
  PATHS "${BFD_ROOT}" PATH_SUFFIXES lib
  REQUIRED)

find_path(TAU_INCLUDE_DIR
  NAMES TAU.h
  PATHS "${TAU_ROOT}" PATH_SUFFIXES include REQUIRED)

find_library(OMP_LIBRARY
  NAMES omp
  PATHS "${OMP_PATH}" REQUIRED)

find_library(TAU_CORE
  NAMES
  tau-phase-ompt-mpi-pdt-openmp
  PATHS "${TAU_ROOT}" PATH_SUFFIXES x86_64/lib REQUIRED)

find_library(TAU_TAUMPI
  NAMES
  TauMpi-phase-ompt-mpi-pdt-openmp
  PATHS "${TAU_ROOT}" PATH_SUFFIXES x86_64/lib REQUIRED)

find_path(UNWIND_INCLUDE_DIR
  NAMES libunwind.h
  PATHS "${UNWIND_ROOT}" PATH_SUFFIXES include
  REQUIRED)

find_library(UNWIND_LIBRARY
  NAMES unwind
  PATHS "${UNWIND_ROOT}" PATH_SUFFIXES lib
  REQUIRED)

find_path(OTF_INCLUDE_DIR
  NAMES otf2/otf2.h
  PATHS "${OTF_ROOT}" PATH_SUFFIXES include
  REQUIRED)

find_library(OTF_LIBRARY
  NAMES otf2
  PATHS "${OTF_ROOT}" PATH_SUFFIXES lib
  REQUIRED)

find_path(DWARF_INCLUDE_DIR
  NAMES dwarf.h
  PATHS "${DWARF_ROOT}" PATH_SUFFIXES include
  REQUIRED)

find_library(DWARF_LIBRARY
  NAMES dwarf
  PATHS "${DWARF_ROOT}" PATH_SUFFIXES lib
  REQUIRED)

find_library(ELF_LIBRARY
  NAMES elf
  PATHS "${DWARF_ROOT}" PATH_SUFFIXES lib
  REQUIRED)

find_package_handle_standard_args(TAU 
  DEFAULT_MSG 
  BFD_LIB
  IBERTY_INCLUDE_DIR
  IBERTY_LIB
  TAU_INCLUDE_DIR
  OMP_LIBRARY
  TAU_CORE
  TAU_TAUMPI
  UNWIND_INCLUDE_DIR
  UNWIND_LIBRARY
  OTF_INCLUDE_DIR
  OTF_LIBRARY
  DWARF_INCLUDE_DIR
  DWARF_LIBRARY
  ELF_LIBRARY)

if (TAU_FOUND)
  add_library(TAU::iberty STATIC IMPORTED)
  set_target_properties(TAU::iberty PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${IBERTY_INCLUDE_DIR}
    IMPORTED_LOCATION ${IBERTY_LIB})
  target_link_libraries(TAU::iberty INTERFACE z dl rt)

  add_library(TAU::BFD SHARED IMPORTED)
  set_target_properties(TAU::BFD PROPERTIES
    IMPORTED_LOCATION ${BFD_LIB})
  target_link_libraries(TAU::BFD INTERFACE TAU::iberty)

  add_library(TAU::unwind STATIC IMPORTED)
  set_target_properties(TAU::unwind PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${UNWIND_INCLUDE_DIR}
    IMPORTED_LOCATION ${UNWIND_LIBRARY})

  add_library(TAU::OTF2 STATIC IMPORTED)
  set_target_properties(TAU::OTF2 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${OTF_INCLUDE_DIR}
    IMPORTED_LOCATION ${OTF_LIBRARY})

  add_library(TAU::elf SHARED IMPORTED)
  set_target_properties(TAU::elf PROPERTIES
    IMPORTED_LOCATION ${ELF_LIBRARY})

  add_library(TAU::DWARF SHARED IMPORTED)
  set_target_properties(TAU::DWARF PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${DWARF_INCLUDE_DIR}
    IMPORTED_LOCATION ${DWARF_LIBRARY})

  add_library(TAU::OpenMP UNKNOWN IMPORTED)
  set_target_properties(TAU::OpenMP PROPERTIES
    IMPORTED_LOCATION ${OMP_LIBRARY})
  target_link_options(TAU::OpenMP INTERFACE -fopenmp)

  add_library(TAU::core UNKNOWN IMPORTED)
  set_target_properties(TAU::core PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES ${TAU_INCLUDE_DIR}
    IMPORTED_LOCATION ${TAU_CORE})
  target_link_libraries(TAU::core INTERFACE TAU::BFD TAU::OpenMP)

  add_library(TAU::TauMpi UNKNOWN IMPORTED)
  set_target_properties(TAU::TauMpi PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES ${TAU_INCLUDE_DIR}
    IMPORTED_LOCATION ${TAU_TAUMPI})
  target_link_libraries(TAU::TauMpi INTERFACE TAU::core MPI::MPI_CXX)

  add_library(TAU::TAU INTERFACE IMPORTED)
  set_property(TARGET TAU::TAU PROPERTY
    INTERFACE_LINK_LIBRARIES ${TAU_LIBS} ${OpenMP_CXX_FLAGS})
  set_property(TARGET TAU::TAU PROPERTY
    INTERFACE_COMPILE_DEFINITIONS ${TAU_COMPILE_DEFS})
  target_compile_options(TAU::TAU INTERFACE -g)
  # Make executable symbols available to dlopen'ed libs
  target_link_options(TAU::TAU INTERFACE LINKER:--export-dynamic)
else()
  message(FATAL "TAU not found")
endif ()
