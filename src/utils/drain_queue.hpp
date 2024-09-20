#pragma once

#include <list>
#include <memory>
#include <mpi.h>

#include "globals.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {
struct SendDrainQueueElement {
  std::shared_ptr<MPI_Request> request;
  int nrefchecks; // Number of checks for use_count = 1
  int ntests;     // Number of MPI_Tests
};

class SendDrainQueue {
  // Collect std::shared_ptr<MPI_Request> objects

public:
  void AddSendRequest(std::shared_ptr<MPI_Request> request) {
    SendDrainQueueElement elem;
    elem.request = request;
    elem.nrefchecks = 0;
    elem.ntests = 0;

    queue_.push_back(elem);
  };

  void TryDraining() {
    for (auto it = queue_.begin(); it != queue_.end();) {
      auto elem = *it;
      if (elem.request.use_count() > 1) {
        elem.nrefchecks++;
        if (elem.nrefchecks > 10) {
          std::cout << "[DrainQueue] use_count > 1 for " << elem.nrefchecks << " tries!"
                    << std::endl;
          continue;
        }
      }

      int flag;
      MPI_Status status;
      PARTHENON_MPI_CHECK(MPI_Test(elem.request.get(), &flag, &status));

      if (flag) {
        it = queue_.erase(it);
      } else {
        elem.ntests++;
        if (elem.ntests > 10) {
          std::cout << "[DrainQueue] MPI_Test failed for " << elem.ntests << " tries!"
                    << std::endl;
          continue;
        }

        it++;
      }
    }
  };

  void ForceDrain() {
    std::cout << "[DrainQueue] Forcing drain of " << queue_.size() << " requests at rank "
              << Globals::my_rank << std::endl;

    for (auto it = queue_.begin(); it != queue_.end();) {
      auto elem = *it;
      MPI_Status status;
      PARTHENON_MPI_CHECK(MPI_Wait(elem.request.get(), &status));
    }
  };

  ~SendDrainQueue() {
    if (queue_.size() > 0) {
      std::cout << "[DrainQueue] Destructing with " << queue_.size()
                << " requests at rank " << Globals::my_rank << std::endl;
      ForceDrain();
    }

    if (queue_.size() > 0) {
      std::stringstream msg;
    }
  };

 private:
  std::list<SendDrainQueueElement> queue_;
};
} // namespace parthenon
