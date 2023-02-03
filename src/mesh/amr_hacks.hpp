//
// Created by Ankush J on 1/19/23.
//

#pragma once

#include <math.h>
#include <numeric>

namespace parthenon {
class AmrHacks {
 public:
  static void AssignBlocks(std::vector<double> const &costlist,
                           std::vector<int> &ranklist) {
    // AssignBlocksRoundRobin(costlist, ranklist);
    AssignBlocksContiguous(costlist, ranklist);
  }

  static int GetLid(std::vector<std::vector<int>> const &rblist,
                    std::vector<int> const &gid2rank, int gid) {
    if (gid >= gid2rank.size()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in GetLID" << std::endl
          << "GID > gid2rank.size() ( " << gid << ", " << gid2rank.size() << ")"
          << std::endl;
      PARTHENON_FAIL(msg);
      return -1;
    }

    int rank = gid2rank[gid];
    if (rank >= rblist.size()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in GetLID" << std::endl
          << "rank >= rblist.size() ( " << rank << ", " << rblist.size() << ")"
          << std::endl;
      PARTHENON_FAIL(msg);
      return -1;
    }

    const std::vector<int> &v = rblist[rank];
    auto it = std::find(v.begin(), v.end(), gid);

    if (it == v.end()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in GetLID" << std::endl
          << "GID " << gid << " could not be located (Rank: " << rank << ")" << std::endl;
      PARTHENON_FAIL(msg);
    }

    int lid = it - v.begin();
    return lid;
  }

 private:
  static void AssignBlocksRoundRobin(std::vector<double> const &costlist,
                                     std::vector<int> &ranklist) {
    int nblocks = costlist.size();
    int nranks = Globals::nranks;

    for (int bid = 0; bid < nblocks; bid++) {
      int bl_rank = bid % nranks;
      ranklist[bid] = bl_rank;
    }

    return;
  }

  static void AssignBlocksSkewed(std::vector<double> const &costlist,
                                 std::vector<int> &ranklist) {
    int nblocks = costlist.size();
    int nranks = Globals::nranks;

    float avg_alloc = nblocks * 1.0f / nranks;
    int rank0_alloc = ceilf(avg_alloc);

    while ((nblocks - rank0_alloc) % (nranks - 1)) {
      rank0_alloc++;
    }

    if (rank0_alloc >= nblocks) {
      std::stringstream msg;
      msg << "### FATAL ERROR rank0_alloc >= nblocks "
          << "(" << rank0_alloc << ", " << nblocks << ")" << std::endl;
      PARTHENON_FAIL(msg);
    }

    for (int bid = 0; bid < nblocks; bid++) {
      if (bid <= rank0_alloc) {
        ranklist[bid] = 0;
      } else {
        int rem_alloc = (nblocks - rank0_alloc) / (nranks - 1);
        int bid_adj = bid - rank0_alloc;
        ranklist[bid] = 1 + bid_adj / rem_alloc;
      }
    }

    return;
  }

  static void AssignBlocksContiguous(std::vector<double> const &costlist,
                                     std::vector<int> &ranklist) {
    ranklist.resize(costlist.size());

    double const total_cost = std::accumulate(costlist.begin(), costlist.end(), 0.0);

    int rank = (Globals::nranks)-1;
    double target_cost = total_cost / Globals::nranks;
    double my_cost = 0.0;
    double remaining_cost = total_cost;
    // create rank list from the end: the master MPI rank should have less load
    for (int block_id = costlist.size() - 1; block_id >= 0; block_id--) {
      if (target_cost == 0.0) {
        std::stringstream msg;
        msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
            << "There is at least one process which has no MeshBlock" << std::endl
            << "Decrease the number of processes or use smaller MeshBlocks." << std::endl;
        PARTHENON_FAIL(msg);
      }
      my_cost += costlist[block_id];
      ranklist[block_id] = rank;
      if (my_cost >= target_cost && rank > 0) {
        rank--;
        remaining_cost -= my_cost;
        my_cost = 0.0;
        target_cost = remaining_cost / (rank + 1);
      }
    }
  }
};
} // namespace parthenon
