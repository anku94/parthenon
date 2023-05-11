#ifndef UTILS_DEBUG_UTILS_HPP_
#define UTILS_DEBUG_UTILS_HPP_
//! \file debug_utils.hpp
//  \brief Debug Utils

#include <stdio.h>
#include <vector>

#include "mesh/meshblock.hpp"

#define LOG_ERRO 5
#define LOG_WARN 4
#define LOG_INFO 3
#define LOG_DBUG 2
#define LOG_DBG2 1

#define LOG_ONLYR0 (1 << 4)

#define LOGR0_ERRO (LOG_ERRO) | (LOG_ONLYR0)
#define LOGR0_WARN (LOG_WARN) | (LOG_ONLYR0)
#define LOGR0_INFO (LOG_INFO) | (LOG_ONLYR0)

/* only levels >= LOG_LEVEL are printed */
#ifndef LOG_LEVEL
#define LOG_LEVEL LOG_INFO
#endif

namespace parthenon {
class DebugUtils {
 public:
  static void Log(int lvl, const char *fmt, ...);
  static void LogBlockList(const char *prefix_str, BlockList_t &block_list);
  static void LogFunc(int bid, const char *func_name, const char *usage_id, int peer,
                      int buf_id, void *ptr);

  static void LogRankBlockList(std::vector<std::vector<int>> const &rblist,
                               std::vector<int> const &nblist);
  template<typename T>
  static void LogArray(int rank, const char* prefix, T* arr, int n);
};
} // namespace parthenon

#endif // UTILS_DEBUG_UTILS_HPP_
