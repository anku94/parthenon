#include "debug_utils.hpp"

#include "globals.hpp"

namespace parthenon {
void DebugUtils::Log(int lvl, const char *fmt, ...) {
  if ((lvl & 7) < LOG_LEVEL) return;
  if ((lvl & LOG_ONLYR0) && (Globals::my_rank != 0)) {
    return;
  }

  const char *prefix;
  va_list ap;
  switch (lvl) {
  case LOG_ERRO:
    prefix = "!!! ERROR !!! ";
    break;
  case LOG_WARN:
    prefix = "-WARNING- ";
    break;
  case LOG_INFO:
    prefix = "-INFO- ";
    break;
  case LOG_DBUG:
    prefix = "-DEBUG- ";
    break;
  default:
    prefix = "";
    break;
  }
  fprintf(stderr, "%s", prefix);

  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);

  fprintf(stderr, "\n");
}

void DebugUtils::LogFunc(int bid, const char *func_name, const char *usage_id, int peer,
                         int buf_id, void *ptr) {
  return;
  int us = Globals::my_rank;
  Log(LOG_DBG2, "%d,%s,%s,%d,%d,%d,%p\n", bid, func_name, usage_id, us, peer, buf_id,
      ptr);
}

void DebugUtils::LogRankBlockList(std::vector<std::vector<int>> const &rblist,
                                   std::vector<int> const &nblist) {
  std::stringstream ss_log;
  for (int r = 0; r < rblist.size(); r++) {
    ss_log << "Rank " << r << ": ";
    for (int bid : rblist[r]) {
      ss_log << bid << ", ";
    }
    ss_log << std::endl;
  }

  Log(LOG_DBUG | LOG_ONLYR0, "[Rank %d] Block Assignment: \n%s", Globals::my_rank,
      ss_log.str().c_str());
}

void DebugUtils::LogBlockList(const char *prefix_str, BlockList_t &block_list) {
  std::stringstream ss_log;
  for (int bidx = 0; bidx < block_list.size(); bidx++) {
    int gid = block_list[bidx]->gid;
    int lid = block_list[bidx]->lid;
    ss_log << "(" << gid << ", " << lid << "), ";
  }

  Log(LOG_DBUG, "[Rank %d] %s: \n%s\n", parthenon::Globals::my_rank, prefix_str,
      ss_log.str().c_str());
}

template<typename T>
void DebugUtils::LogArray(int rank, const char* prefix, T* arr, int n) {
  std::stringstream ss;
  ss << "[Rank " << rank << "] " << prefix;
  ss << "(" << n << " items): ";

  for (int i = 0; i < n; i++) {
    ss << arr[i] << ", ";
  }

  Log(LOG_INFO, "%s", ss.str().c_str());

}

template void DebugUtils::LogArray(int rank, const char* prefix, int* arr, int n);
template void DebugUtils::LogArray(int rank, const char* prefix, double* arr, int n);
} // namespace parthenon
