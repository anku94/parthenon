#pragma once

#include "globals.hpp"

#include <Profile/TauPluginTypes.h>
#include <TAU.h>

#define TAU_BLKEVT_US_FD 0
#define TAU_BLKEVT_US_CF 1
#define TAU_BLKEVT_FLAG_REF 2
#define TAU_BLKEVT_RANK_GID 3

#define TAUPROF_ENABLE 1

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
