//
// Created by Ankush J on 1/19/23.
//

#pragma once

namespace parthenon {
class AmrHacks {
 public:
  static void AssignBlocks(std::vector<double> const &costlist,
                           std::vector<int> &ranklist) {
    return AssignBlocksRoundRobin(costlist, ranklist);
  }

  static int GetLid(std::vector<std::vector<int>> const &rblist,
                    std::vector<int> const &gid2rank, int gid) {
    if (gid >= gid2rank.size()) return -1;

    int rank = gid2rank[gid];
    if (rank >= rblist.size()) return -1;

    const std::vector<int> &v = rblist[rank];
    auto it = std::find(v.begin(), v.end(), gid);

    return (it == v.end()) ? -1 : (it - v.begin());
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
    int rank0_alloc = std::ceilf(avg_alloc);

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
};
} // namespace parthenon