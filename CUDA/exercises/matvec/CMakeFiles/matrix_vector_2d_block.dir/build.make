# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sstaccone/exercises

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sstaccone/exercises

# Include any dependencies generated for this target.
include matvec/CMakeFiles/matrix_vector_2d_block.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include matvec/CMakeFiles/matrix_vector_2d_block.dir/compiler_depend.make

# Include the progress variables for this target.
include matvec/CMakeFiles/matrix_vector_2d_block.dir/progress.make

# Include the compile flags for this target's objects.
include matvec/CMakeFiles/matrix_vector_2d_block.dir/flags.make

matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o: matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o.depend
matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o: matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o.cmake
matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o: matvec/03-matrix_vector_2d_block.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/sstaccone/exercises/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o"
	cd /home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir && /usr/bin/cmake -E make_directory /home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir//.
	cd /home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir//./matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o -D generated_cubin_file:STRING=/home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir//./matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o.cubin.txt -P /home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir//matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o.cmake

# Object files for target matrix_vector_2d_block
matrix_vector_2d_block_OBJECTS =

# External object files for target matrix_vector_2d_block
matrix_vector_2d_block_EXTERNAL_OBJECTS = \
"/home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o"

matvec/matrix_vector_2d_block: matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o
matvec/matrix_vector_2d_block: matvec/CMakeFiles/matrix_vector_2d_block.dir/build.make
matvec/matrix_vector_2d_block: /opt/cuda/12.3.0/lib64/libcudart_static.a
matvec/matrix_vector_2d_block: /usr/lib64/librt.a
matvec/matrix_vector_2d_block: matvec/CMakeFiles/matrix_vector_2d_block.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sstaccone/exercises/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable matrix_vector_2d_block"
	cd /home/sstaccone/exercises/matvec && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_vector_2d_block.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
matvec/CMakeFiles/matrix_vector_2d_block.dir/build: matvec/matrix_vector_2d_block
.PHONY : matvec/CMakeFiles/matrix_vector_2d_block.dir/build

matvec/CMakeFiles/matrix_vector_2d_block.dir/clean:
	cd /home/sstaccone/exercises/matvec && $(CMAKE_COMMAND) -P CMakeFiles/matrix_vector_2d_block.dir/cmake_clean.cmake
.PHONY : matvec/CMakeFiles/matrix_vector_2d_block.dir/clean

matvec/CMakeFiles/matrix_vector_2d_block.dir/depend: matvec/CMakeFiles/matrix_vector_2d_block.dir/matrix_vector_2d_block_generated_03-matrix_vector_2d_block.cu.o
	cd /home/sstaccone/exercises && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sstaccone/exercises /home/sstaccone/exercises/matvec /home/sstaccone/exercises /home/sstaccone/exercises/matvec /home/sstaccone/exercises/matvec/CMakeFiles/matrix_vector_2d_block.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : matvec/CMakeFiles/matrix_vector_2d_block.dir/depend

