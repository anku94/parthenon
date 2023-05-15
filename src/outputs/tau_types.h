#pragma once

#include "globals.hpp"
#include <vector>

#ifdef TAUPROF_ENABLE
#include <Profile/TauPluginTypes.h>
#include <TAU.h>
#endif

#define TAU_BLKEVT_US_FD 0
#define TAU_BLKEVT_US_CF 1
#define TAU_BLKEVT_FLAG_REF 2
#define TAU_BLKEVT_RANK_GID 3

namespace tau {

enum class MsgType { kBlockAssignment, kTargetCost, kTsEnd, kBlockEvent };

struct TriggerMsg {
  MsgType type;
  void *data;
};

struct MsgBlockAssignment {
  std::vector<double> const *costlist;
  std::vector<int> const *ranklist;
};

struct MsgBlockEvent {
  int block_id;
  int opcode;
  int data;
};

double GetUsSince(double since);

void LogBlockAssignment(std::vector<double> const &costlist, std::vector<int> &ranklist);

void LogTargetCost(double cost);

void MarkTimestepEnd();

void LogBlockEvent(int block_id, int opcode, int data);
} // namespace tau
