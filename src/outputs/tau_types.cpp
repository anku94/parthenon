#include "tau_types.hpp"

#include <time.h>

#define TAUPROF_ENABLE 1

namespace tau {
double GetUsSince(double since) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  double cur_us = t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
  return cur_us - since;
}

void LogBlockAssignment(std::vector<double> const &costlist, std::vector<int> &ranklist) {
#if TAUPROF_ENABLE == 1
  MsgBlockAssignment msg;
  msg.costlist = &costlist;
  msg.ranklist = &ranklist;

  TriggerMsg tmsg;
  tmsg.type = MsgType::kBlockAssignment;
  tmsg.data = (void *)&msg;

  TAU_PROFILE_TIMER(timer, "trigger_timer", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
  TAU_PROFILE_STOP(timer);
#endif
}

void LogTargetCost(double cost) {
#if TAUPROF_ENABLE == 1
  TriggerMsg tmsg;
  tmsg.type = MsgType::kTargetCost;
  tmsg.data = (void *)&cost;

  TAU_PROFILE_TIMER(timer, "trigger_timer", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
  TAU_PROFILE_STOP(timer);
#endif
}

void MarkTimestepEnd() {
#if TAUPROF_ENABLE == 1
  TriggerMsg tmsg;
  tmsg.type = MsgType::kTsEnd;
  tmsg.data = nullptr;

  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
#endif
}

void LogBlockEvent(int block_id, int opcode, int data) {
#if TAUPROF_ENABLE == 1
  // fprintf(stderr, "LOG EVENT TIME\n");

  MsgBlockEvent msg{block_id, opcode, data};

  TriggerMsg tmsg;
  tmsg.type = MsgType::kBlockEvent;
  tmsg.data = (void *)&msg;

  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
#endif
}

void LogCommChannel(void *ptr, int block_id, int block_rank, int nbr_id, int nbr_rank,
                    int tag, char is_flux) {
#if TAUPROF_ENABLE == 1
  // fprintf(stderr, "LOG EVENT TIME\n");

  MsgCommChannel msg{ptr, block_id, block_rank, nbr_id, nbr_rank, tag, is_flux};

  TriggerMsg tmsg;
  tmsg.type = MsgType::kCommChannel;
  tmsg.data = (void *)&msg;

  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
#endif
}

void LogMsgSend(void *ptr, int buf_sz, int recv_rank, int tag, uint64_t timestamp) {
#if TAUPROF_ENABLE == 1
  MsgSend msg{ptr, buf_sz, recv_rank, tag, timestamp};

  TriggerMsg tmsg;
  tmsg.type = MsgType::kMsgSend;
  tmsg.data = (void *)&msg;

  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
#endif
}
} // namespace tau
