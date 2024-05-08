#pragma once

#include "globals.hpp"
#include <vector>

#undef TAUPROF_ENABLE

#ifdef TAUPROF_ENABLE
#include <Profile/TauPluginTypes.h>
#include <TAU.h>
#endif

#define TAU_BLKEVT_US_FD 0
#define TAU_BLKEVT_US_CF 1
#define TAU_BLKEVT_FLAG_REF 2
#define TAU_BLKEVT_RANK_GID 3
#define TAU_BLKEVT_COST 4
#define TAU_BLKEVT_US_COMP3 5
#define TAU_BLKEVT_US_COMP4 6

namespace tau {

enum class MsgType {
  kBlockAssignment,
  kTargetCost,
  kTsEnd,
  kBlockEvent,
  kCommChannel,
  kMsgSend
};

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

struct MsgCommChannel {
  void *ptr;
  int block_id;
  int block_rank;
  int nbr_id;
  int nbr_rank;
  int tag;
  char is_flux;
};

struct MsgSend {
  void *ptr;
  int buf_sz;
  int recv_rank;
  int tag;
  uint64_t timestamp;
};

double GetUsSince(double since);

void LogBlockAssignment(std::vector<double> const &costlist, std::vector<int> &ranklist);

void LogTargetCost(double cost);

void MarkTimestepEnd();

void LogBlockEvent(int block_id, int opcode, int data);

void LogCommChannel(void *ptr, int block_id, int block_rank, int nbr_id, int nbr_rank,
                    int tag, char is_flux);

void LogMsgSend(void *ptr, int buf_sz, int recv_rank, int tag, uint64_t timestamp);
} // namespace tau
