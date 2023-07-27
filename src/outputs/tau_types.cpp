#include "tau_types.h"

#include <time.h>

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

  MsgBlockEvent msg;
  msg.block_id = block_id;
  msg.opcode = opcode;
  msg.data = data;

  TriggerMsg tmsg;
  tmsg.type = MsgType::kBlockEvent;
  tmsg.data = (void *)&msg;

  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
#endif
}
} // namespace tau
