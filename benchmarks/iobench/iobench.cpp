#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // For getopt

int rank;

void printf_at_zero(const char *format, ...) {
  va_list args;
  va_start(args, format);

  if (rank == 0) {
    vprintf(format, args);
  }

  va_end(args);
}

void get_fname_from_opts(int argc, char **argv, char **fname) {
  int opt;
  while ((opt = getopt(argc, argv, "i:")) != -1) {
    if (opt == 'i') {
      *fname = optarg;
      return;
    }
  }
  *fname = nullptr;
}

int read_int_from_file(MPI_Comm comm, char *filename) {
  MPI_File fh;
  char data[10];
  int value;

  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  MPI_File_read(fh, data, 10, MPI_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
  value = atoi(data);
  return value;
}

void compute_and_print(MPI_Comm comm, int value) {
  int sum;
  MPI_Allreduce(&value, &sum, 1, MPI_INT, MPI_SUM, comm);
  printf_at_zero("Sum: %d\n", sum);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char *filename = nullptr;
  int value = -1;
  get_fname_from_opts(argc, argv, &filename);

  if (filename == nullptr) {
    fprintf(stderr, "Usage: %s -i filename\n", argv[0]);
    goto FINALIZE;
  }

  printf_at_zero("Filename: %s\n", filename);
  value = read_int_from_file(MPI_COMM_WORLD, filename);
  printf_at_zero("Value: %d\n", value);
  compute_and_print(MPI_COMM_WORLD, value);

FINALIZE:
  MPI_Finalize();
  return 0;
}
