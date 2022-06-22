#pragma once

#include "globals.hpp"

#include <TAU.h>
#include <Profile/TauPluginTypes.h>

namespace tau {

enum class MsgType { kBlockAssignment, kTargetCost };

struct TriggerMsg {
  MsgType type;
  void* data;
};

struct MsgBlockAssignment {
  std::vector<double> const *costlist;
  std::vector<int> const* ranklist;
};

void LogBlockAssignment(std::vector<double> const &costlist, std::vector<int> &ranklist) {

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
}

void LogTargetCost(double cost) {
  TriggerMsg tmsg;
  tmsg.type = MsgType::kTargetCost;
  tmsg.data = (void *)&cost;

  TAU_PROFILE_TIMER(timer, "trigger_timer", "", TAU_DEFAULT);
  TAU_PROFILE_START(timer);
  TAU_TRIGGER(parthenon::Globals::tau_amr_module, (void *)&tmsg);
  TAU_PROFILE_STOP(timer);
}

} // namespace tau

