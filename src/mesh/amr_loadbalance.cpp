//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//! \file mesh_amr.cpp
//  \brief implementation of Mesh::AdaptiveMeshRefinement() and related utilities

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>

#include <lb_policies.h>
#include <policy.h>

#include "parthenon_mpi.hpp"

#include "bvals/boundary_conditions.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/update.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_tree.hpp"
#include "outputs/tau_types.hpp"
#include "parthenon_arrays.hpp"
#include "utils/buffer_utils.hpp"
#include "utils/error_checking.hpp"

#include "perfsignal.h"

static PerfManager perf;

namespace parthenon {

void PrintCosts(std::vector<double> const &costlist) {
  if (Globals::my_rank != 0) {
    return;
  }

  std::stringstream ss;
  for (int i = 0; i < costlist.size(); i++) {
    ss << std::fixed << std::setprecision(1) << costlist[i] << " ";
    if (i % 32 == 31) {
      ss << std::endl;
    }
  }

  std::cout << "Costlist (" << costlist.size() << " blocks): " << std::endl
            << ss.str() << std::endl;
}

void ComputeAssignmentStatistics(std::vector<double> const &costlist,
                                 std::vector<int> const &ranklist, int nranks) {
  if (Globals::my_rank != 0) {
    return;
  }

  int nblocks = costlist.size();
  std::vector<double> rank_costs(nranks, 0.0);
  std::vector<double> rank_counts(nranks, 0.0);

  for (int bidx = 0; bidx < nblocks; bidx++) {
    int rank = ranklist[bidx];
    rank_costs[rank] += costlist[bidx];
    rank_counts[rank] += 1.0;
  }

  // compute max, med, min of costs
  std::sort(rank_costs.begin(), rank_costs.end());
  double min_cost = rank_costs.front();
  double max_cost = rank_costs.back();
  double med_cost = rank_costs[nranks / 2];

  // compute max, med, min of counts
  std::sort(rank_counts.begin(), rank_counts.end());
  double min_count = rank_counts.front();
  double max_count = rank_counts.back();
  double med_count = rank_counts[nranks / 2];

  std::cout << "Rank statistics (nblocks=" << nblocks << "):" << std::endl;
#define FLOAT(x) std::fixed << std::setprecision(0) << x / 1e3
  std::cout << "  Costs: min=" << FLOAT(min_cost) << " med=" << FLOAT(med_cost)
            << " max=" << FLOAT(max_cost) << std::endl;
  std::cout << "  Counts: min=" << min_count << " med=" << med_count
            << " max=" << max_count << std::endl;
#undef FLOAT
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin)
// \brief Main function for adaptive mesh refinement

void Mesh::LoadBalancingAndAdaptiveMeshRefinement(int ncycles_over, ParameterInput *pin,
                                                  ApplicationInput *app_in) {
  Kokkos::Profiling::pushRegion("LoadBalancingAndAdaptiveMeshRefinement");

  int bidx = 0;
  for (auto &pmb : block_list) {
    tau::LogBlockEvent(pmb->gid, TAU_BLKEVT_RANK_GID, bidx++);
  }

  int nnew = 0, ndel = 0;

  if (adaptive) {
    UpdateMeshBlockTree(nnew, ndel);
    nbnew += nnew;
    nbdel += ndel;
  }

  lb_flag_ |= lb_automatic_;

  UpdateCostList();

  modified = false;

  // to force pack size == 1 in athena_pk
  if (ncycles_over == 1) {
    if (Globals::my_rank == 0) {
      std::cout << "[ALERT] Changing pack size to 1. Prev: " << default_pack_size_
                << std::endl;
    }

    default_pack_size_ = 1;
    GatherCostListAndCheckBalance();
    Globals::perf.Resume();
    RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal);
    Globals::perf.Pause();
    modified = true;
  } else {
    if (nnew != 0 || ndel != 0) { // at least one (de)refinement happened
      GatherCostListAndCheckBalance();
      Globals::perf.Resume();
      RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal + nnew - ndel);
      Globals::perf.Pause();
      modified = true;
    } else if (lb_flag_ && step_since_lb >= lb_interval_) {
      // if (!GatherCostListAndCheckBalance()) { // load imbalance detected
      // XXX AJ: R&R completely at the interval
      GatherCostListAndCheckBalance();
      Globals::perf.Resume();
      RedistributeAndRefineMeshBlocks(pin, app_in, nbtotal);
      Globals::perf.Pause();
      modified = true;
      // }
      lb_flag_ = false;
    }
  }

  // gid = -1 denotes end of LBandAMR
  tau::LogBlockEvent(-1, TAU_BLKEVT_FLAG_REF, 0);
  Kokkos::Profiling::popRegion(); // LoadBalancingAndAdaptiveMeshRefinement
}

// Private routines
namespace {
/**
 * @brief This routine assigns blocks to ranks by attempting to place index-contiguous
 * blocks of equal total cost on each rank.
 *
 * @param costlist (Input) A map of global block ID to a relative weight.
 * @param ranklist (Output) A map of global block ID to ranks.
 */
void AssignBlocks(std::vector<double> const &costlist, std::vector<int> &ranklist) {
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

void UpdateBlockList(std::vector<int> const &ranklist,
                     std::vector<std::vector<int>> &rblist, std::vector<int> &nblist) {
  rblist.resize(Globals::nranks);
  nblist.resize(Globals::nranks);

  for (auto &v : rblist) {
    v.clear();
  }

  std::fill(nblist.begin(), nblist.end(), 0);

  for (int block_id = 0; block_id < ranklist.size(); block_id++) {
    int rank = ranklist[block_id];
    rblist[rank].push_back(block_id);
    nblist[rank]++;
  }
}
} // namespace

//----------------------------------------------------------------------------------------
// \brief Calculate distribution of MeshBlocks based on the cost list
void Mesh::CalculateLoadBalance(std::vector<double> const &costlist,
                                std::vector<int> &ranklist,
                                std::vector<std::vector<int>> &rblist,
                                std::vector<int> &nblist) {
  Kokkos::Profiling::pushRegion("CalculateLoadBalance");
  auto const total_blocks = costlist.size();

  using it = std::vector<double>::const_iterator;
  std::pair<it, it> const min_max = std::minmax_element(costlist.begin(), costlist.end());

  double const mincost = min_max.first == costlist.begin() ? 0.0 : *min_max.first;
  double const maxcost = min_max.second == costlist.begin() ? 0.0 : *min_max.second;

  if (Globals::my_rank == 0) {
    std::cout << "[LB] " << Globals::lb_policy << " being invoked!" << std::endl;
  }

  // if (ncycles_over == 1) {
  // AssignBlocks(costlist, ranklist);
  // } else {
  if (Globals::nranks < 512) {
    amr::LoadBalancePolicies::AssignBlocks(Globals::lb_policy.c_str(), costlist, ranklist,
                                           Globals::nranks);
  } else {
    amr::LoadBalancePolicies::AssignBlocksParallel(
        Globals::lb_policy.c_str(), costlist, ranklist, Globals::nranks, MPI_COMM_WORLD);
  }
  //
  // if (Globals::my_rank == 0) {
  //   std::cout << "[LB] " << Globals::my_rank << " assigned " << ranklist.size()
  //             << " blocks, last rank: " << ranklist.back() << std::endl;
  // }
  ComputeAssignmentStatistics(costlist, ranklist, Globals::nranks);

  // }

  // Updates nslist with the ID of the starting block on each rank and the count of blocks
  // on each rank.
  UpdateBlockList(ranklist, rblist, nblist);

#ifdef MPI_PARALLEL
  if (total_blocks % (Globals::nranks) != 0 && !adaptive && !lb_flag_ &&
      maxcost == mincost && Globals::my_rank == 0) {
    std::cout << "### Warning in CalculateLoadBalance" << std::endl
              << "The number of MeshBlocks cannot be divided evenly. "
              << "This will result in poor load balancing." << std::endl;
  }
#endif
  if (Globals::nranks > total_blocks) {
    if (!adaptive) {
      // mesh is refined statically, treat this an as error (all ranks need to
      // participate)
      std::stringstream msg;
      msg << "### FATAL ERROR in CalculateLoadBalance" << std::endl
          << "There are fewer MeshBlocks than OpenMP threads on each MPI rank"
          << std::endl
          << "Decrease the number of threads or use more MeshBlocks." << std::endl;
      PARTHENON_FAIL(msg);
    } else if (Globals::my_rank == 0) {
      // we have AMR, print warning only on Rank 0
      std::cout << "### WARNING in CalculateLoadBalance" << std::endl
                << "There are fewer MeshBlocks than OpenMP threads on each MPI rank"
                << std::endl
                << "This is likely fine if the number of meshblocks is expected to grow "
                   "during the "
                   "simulations. Otherwise, it might be worthwhile to decrease the "
                   "number of threads or "
                   "use more meshblocks."
                << std::endl;
    }
  }
  Kokkos::Profiling::popRegion(); // CalculateLoadBalance
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::ResetLoadBalanceVariables()
// \brief reset counters and flags for load balancing

void Mesh::ResetLoadBalanceVariables() {
  if (lb_automatic_) {
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = TINY_NUMBER;
      pmb->ResetTimeMeasurement();
    }
  } else if (lb_manual_) {
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = TINY_NUMBER;
      // pmb->ResetCostForLoadBalancing();
    }
  }
  lb_flag_ = false;
  step_since_lb = 0;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateCostList()
// \brief update the cost list

void Mesh::UpdateCostList() {
  if (lb_automatic_) {
    double w = static_cast<double>(lb_interval_ - 1) / static_cast<double>(lb_interval_);
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = costlist[pmb->gid] * w + pmb->cost_;
    }
  } else if (lb_flag_) {
    for (auto &pmb : block_list) {
      costlist[pmb->gid] = pmb->cost_;
      tau::LogBlockEvent(pmb->gid, TAU_BLKEVT_COST, pmb->cost_);
    }
  }
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel)
// \brief collect refinement flags and manipulate the MeshBlockTree

void Mesh::UpdateMeshBlockTree(int &nnew, int &ndel) {
  Kokkos::Profiling::pushRegion("UpdateMeshBlockTree");
  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (mesh_size.nx2 > 1) nleaf = 4;
  if (mesh_size.nx3 > 1) nleaf = 8;

  // collect refinement flags from all the meshblocks
  // count the number of the blocks to be (de)refined
  nref[Globals::my_rank] = 0;
  nderef[Globals::my_rank] = 0;
  for (auto const &pmb : block_list) {
    if (pmb->pmr->refine_flag_ == 1) nref[Globals::my_rank]++;
    if (pmb->pmr->refine_flag_ == -1) nderef[Globals::my_rank]++;
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nref.data(), 1, MPI_INT, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, nderef.data(), 1, MPI_INT, MPI_COMM_WORLD));
#endif

  // count the number of the blocks to be (de)refined and displacement
  int tnref = 0, tnderef = 0;
  for (int n = 0; n < Globals::nranks; n++) {
    tnref += nref[n];
    tnderef += nderef[n];
  }
  if (tnref == 0 && tnderef < nleaf) { // nothing to do
    Kokkos::Profiling::popRegion();    // UpdateMeshBlockTree
    return;
  }

  int rd = 0, dd = 0;
  for (int n = 0; n < Globals::nranks; n++) {
    rdisp[n] = rd;
    ddisp[n] = dd;
    // technically could overflow, since sizeof() operator returns
    // std::size_t = long unsigned int > int
    // on many platforms (LP64). However, these are used below in MPI calls for
    // integer arguments (recvcounts, displs). MPI does not support > 64-bit count ranges
    bnref[n] = static_cast<int>(nref[n] * sizeof(LogicalLocation));
    bnderef[n] = static_cast<int>(nderef[n] * sizeof(LogicalLocation));
    brdisp[n] = static_cast<int>(rd * sizeof(LogicalLocation));
    bddisp[n] = static_cast<int>(dd * sizeof(LogicalLocation));
    rd += nref[n];
    dd += nderef[n];
  }

  // allocate memory for the location arrays
  LogicalLocation *lref{}, *lderef{}, *clderef{};
  if (tnref > 0) lref = new LogicalLocation[tnref];
  if (tnderef >= nleaf) {
    lderef = new LogicalLocation[tnderef];
    clderef = new LogicalLocation[tnderef / nleaf];
  }

  // collect the locations and costs
  int iref = rdisp[Globals::my_rank], ideref = ddisp[Globals::my_rank];
  for (auto const &pmb : block_list) {
    if (pmb->pmr->refine_flag_ == 1) lref[iref++] = pmb->loc;
    if (pmb->pmr->refine_flag_ == -1 && tnderef >= nleaf) lderef[ideref++] = pmb->loc;
  }
#ifdef MPI_PARALLEL
  if (tnref > 0) {
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, bnref[Globals::my_rank], MPI_BYTE,
                                       lref, bnref.data(), brdisp.data(), MPI_BYTE,
                                       MPI_COMM_WORLD));
  }
  if (tnderef >= nleaf) {
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, bnderef[Globals::my_rank], MPI_BYTE,
                                       lderef, bnderef.data(), bddisp.data(), MPI_BYTE,
                                       MPI_COMM_WORLD));
  }
#endif

  if (lref) {
    std::sort(lref, lref + tnref, LogicalLocation::SortComparator);
  }

  if (lderef) {
    std::sort(lderef, lderef + tnderef, LogicalLocation::SortComparator);
  }

  // calculate the list of the newly derefined blocks
  int ctnd = 0;
  if (tnderef >= nleaf) {
    int lk = 0, lj = 0;
    if (mesh_size.nx2 > 1) lj = 1;
    if (mesh_size.nx3 > 1) lk = 1;
    for (int n = 0; n < tnderef; n++) {
      if ((lderef[n].lx1 & 1LL) == 0LL && (lderef[n].lx2 & 1LL) == 0LL &&
          (lderef[n].lx3 & 1LL) == 0LL) {
        int r = n, rr = 0;
        for (std::int64_t k = 0; k <= lk; k++) {
          for (std::int64_t j = 0; j <= lj; j++) {
            for (std::int64_t i = 0; i <= 1; i++) {
              if (r < tnderef) {
                if ((lderef[n].lx1 + i) == lderef[r].lx1 &&
                    (lderef[n].lx2 + j) == lderef[r].lx2 &&
                    (lderef[n].lx3 + k) == lderef[r].lx3 &&
                    lderef[n].level == lderef[r].level)
                  rr++;
                r++;
              }
            }
          }
        }
        if (rr == nleaf) {
          clderef[ctnd].lx1 = lderef[n].lx1 >> 1;
          clderef[ctnd].lx2 = lderef[n].lx2 >> 1;
          clderef[ctnd].lx3 = lderef[n].lx3 >> 1;
          clderef[ctnd].level = lderef[n].level - 1;
          ctnd++;
        }
      }
    }
  }
  // sort the lists by level
  if (ctnd > 1) std::sort(clderef, &(clderef[ctnd - 1]), LogicalLocation::Greater);

  if (tnderef >= nleaf) delete[] lderef;

  // Now the lists of the blocks to be refined and derefined are completed
  // Start tree manipulation
  // Step 1. perform refinement
  for (int n = 0; n < tnref; n++) {
    MeshBlockTree *bt = tree.FindMeshBlock(lref[n]);
    bt->Refine(nnew);
  }
  if (tnref != 0) delete[] lref;

  // Step 2. perform derefinement
  for (int n = 0; n < ctnd; n++) {
    MeshBlockTree *bt = tree.FindMeshBlock(clderef[n]);
    bt->Derefine(ndel);
  }
  if (tnderef >= nleaf) delete[] clderef;

  Kokkos::Profiling::popRegion(); // UpdateMeshBlockTree
}

//----------------------------------------------------------------------------------------
// \!fn bool Mesh::GatherCostListAndCheckBalance()
// \brief collect the cost from MeshBlocks and check the load balance

bool Mesh::GatherCostListAndCheckBalance() {
  if (lb_manual_ || lb_automatic_) {
    // local data structure, start offsets for costlist_ro
    int nslist_loc[Globals::nranks];
    // temporary costlist, rank ordered. global costlist is gid-ordered
    double costlist_loc[nbtotal];

    // Setup nslist
    nslist_loc[0] = 0;
    for (int ridx = 1; ridx < Globals::nranks; ridx++) {
      nslist_loc[ridx] = nslist_loc[ridx - 1] + nblist[ridx - 1];
    }

    // Setup costlist_ro from Mesh::costlist
    int nblocal = nblist[Globals::my_rank];
    int nbs = nslist_loc[Globals::my_rank];

    const std::vector<int> &myrblist = rblist[Globals::my_rank];

    for (int bidx = 0; bidx < nblocal; bidx++) {
      int bid = myrblist[bidx];
      costlist_loc[nbs + bidx] = costlist[bid];
    }

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_DOUBLE,
                                       costlist_loc, nblist.data(), nslist_loc,
                                       MPI_DOUBLE, MPI_COMM_WORLD));
#endif
    double maxcost = 0.0, avecost = 0.0;
    double mincost = 1e100;
    for (int rank = 0; rank < Globals::nranks; rank++) {
      double rcost = 0.0;
      int ns = nslist_loc[rank];
      int ne = ns + nblist[rank];
      for (int n = ns; n < ne; ++n)
        rcost += costlist_loc[n];
      maxcost = std::max(maxcost, rcost);
      avecost += rcost;
      mincost = std::min(mincost, rcost);
    }
    avecost /= Globals::nranks;

    if (Globals::my_rank == 0) {
      fprintf(stdout, "[LB] Max: %.1f Avg: %.1f Min: %.1f\n", maxcost / 1e6,
              avecost / 1e6, mincost / 1e6);
    }

    // Copy costlist_ro back to Mesh::costlist
    // Currently copying the full costlist.
    // Possible that only inner loop for ridx == Global::my_rank is needed

    for (int ridx = 0; ridx < Globals::nranks; ridx++) {
      int nsrank = nslist_loc[ridx];
      for (int bidx = 0; bidx < nblist[ridx]; bidx++) {
        int bid = rblist[ridx][bidx];
        costlist[bid] = costlist_loc[nsrank + bidx];
      }
    }

    // PrintCosts(costlist);

    if (adaptive)
      lb_tolerance_ =
          2.0 * static_cast<double>(Globals::nranks) / static_cast<double>(nbtotal);

    if (maxcost > (1.0 + lb_tolerance_) * avecost) return false;
  }
  return true;
}

//----------------------------------------------------------------------------------------
// \!fn void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, int ntot)
// \brief redistribute MeshBlocks according to the new load balance

void Mesh::RedistributeAndRefineMeshBlocks(ParameterInput *pin, ApplicationInput *app_in,
                                           int ntot) {
  perf.Resume();

  auto _ts_beg = tau::GetUsSince(0);

  Kokkos::Profiling::pushRegion("RedistributeAndRefineMeshBlocks");
  // kill any cached packs
  mesh_data.PurgeNonBase();
  mesh_data.Get()->ClearCaches();

  // compute nleaf= number of leaf MeshBlocks per refined block
  int nleaf = 2;
  if (mesh_size.nx2 > 1) nleaf = 4;
  if (mesh_size.nx3 > 1) nleaf = 8;

  // Step 1. construct new lists
  Kokkos::Profiling::pushRegion("Step1: Construct new list");
  std::vector<LogicalLocation> newloc(ntot);
  std::vector<int> newrank(ntot);
  std::vector<double> newcost(ntot);
  std::vector<int> newtoold(ntot);
  std::vector<int> oldtonew(nbtotal);

  int nbtold = nbtotal;
  tree.GetMeshBlockList(newloc.data(), newtoold.data(), nbtotal);

  // create a list mapping the previous gid to the current one
  oldtonew[0] = 0;
  int mb_idx = 1;
  for (int n = 1; n < ntot; n++) {
    if (newtoold[n] == newtoold[n - 1] + 1) { // normal
      oldtonew[mb_idx++] = n;
    } else if (newtoold[n] == newtoold[n - 1] + nleaf) { // derefined
      for (int j = 0; j < nleaf - 1; j++)
        oldtonew[mb_idx++] = n - 1;
      oldtonew[mb_idx++] = n;
    }
  }
  // fill the last block
  for (; mb_idx < nbtold; mb_idx++)
    oldtonew[mb_idx] = ntot - 1;

  current_level = 0;
  for (int n = 0; n < ntot; n++) {
    // "on" = "old n" = "old gid" = "old global MeshBlock ID"
    int on = newtoold[n];
    if (newloc[n].level > current_level) // set the current max level
      current_level = newloc[n].level;
    if (newloc[n].level >= loclist[on].level) { // same or refined
      newcost[n] = costlist[on];
    } else {
      double acost = 0.0;
      for (int l = 0; l < nleaf; l++)
        acost += costlist[on + l];
      newcost[n] = acost / nleaf;
    }
  }
#ifdef MPI_PARALLEL
  // store old nbstart and nbend before load balancing in Step 2.
  std::vector<int> omyrblist(nblist[Globals::my_rank]);
  std::copy(rblist[Globals::my_rank].begin(), rblist[Globals::my_rank].end(),
            omyrblist.begin());
#endif
  Kokkos::Profiling::popRegion(); // Step 1

  // Step 2. Calculate new load balance
  CalculateLoadBalance(newcost, newrank, rblist, nblist);

  std::vector<int> &nmyrblist = rblist[Globals::my_rank];

#ifdef MPI_PARALLEL
  int bnx1 = GetBlockSize().nx1;
  int bnx2 = GetBlockSize().nx2;
  int bnx3 = GetBlockSize().nx3;
  // Step 3. count the number of the blocks to be sent / received
  Kokkos::Profiling::pushRegion("Step 3: Count blocks");
  int nsend = 0, nrecv = 0;
  for (int n : nmyrblist) {
    int on = newtoold[n];
    if (loclist[on].level > newloc[n].level) { // f2c
      for (int k = 0; k < nleaf; k++) {
        if (ranklist[on + k] != Globals::my_rank) nrecv++;
      }
    } else {
      if (ranklist[on] != Globals::my_rank) nrecv++;
    }
  }
  for (int n : omyrblist) {
    int nn = oldtonew[n];
    if (loclist[n].level < newloc[nn].level) { // c2f
      for (int k = 0; k < nleaf; k++) {
        if (newrank[nn + k] != Globals::my_rank) nsend++;
      }
    } else {
      if (newrank[nn] != Globals::my_rank) nsend++;
    }
  }

  Kokkos::Profiling::popRegion(); // Step 3
  // Step 4. calculate buffer sizes
  Kokkos::Profiling::pushRegion("Step 4: Calc buffer sizes");
  BufArray1D<Real> *sendbuf, *recvbuf;
  // use the first MeshBlock in the linked list of blocks belonging to this MPI rank as a
  // representative of all MeshBlocks for counting the "load-balancing registered" and
  // "SMR/AMR-enrolled" quantities (loop over MeshBlock::vars_cc_, not MeshRefinement)

  // TODO(felker): add explicit check to ensure that elements of pb->vars_cc/fc_ and
  // pb->pmr->pvars_cc/fc_ v point to the same objects, if adaptive

  // TODO(JL) Why are we using all variables for same-level but only the variables in pmr
  // for c2f and f2c?s
  int num_cc = block_list.front()->vars_cc_.size();
  int num_pmr_cc = block_list.front()->pmr->pvars_cc_.size();
  int nx4_tot = 0;
  for (auto &pvar_cc : block_list.front()->vars_cc_) {
    nx4_tot += pvar_cc->GetDim(4);
  }

  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  // cell-centered quantities enrolled in SMR/AMR
  // TODO(JMM): I think this needs to be re-written to compute total
  // size accross vars by looping over vars and getting their total
  // size.
  int bssame = bnx1 * bnx2 * bnx3 * nx4_tot;
  int bsf2c = (bnx1 / 2) * ((bnx2 + 1) / 2) * ((bnx3 + 1) / 2) * nx4_tot;
  int bsc2f =
      (bnx1 / 2 + 2) * ((bnx2 + 1) / 2 + 2 * f2) * ((bnx3 + 1) / 2 + 2 * f3) * nx4_tot;

  // add num_cc/num_pmr_cc to all buffer sizes for storing allocation statuses
  bssame += num_cc;
  bsc2f += num_pmr_cc;
  bsf2c += num_pmr_cc;

  // add one more element to buffer size for storing the derefinement counter
  bssame++;
  Kokkos::Profiling::popRegion(); // Step 4

  MPI_Request *req_send, *req_recv;

  // Step 5. Allocate space for send and recieve buffers
  Kokkos::Profiling::pushRegion("Step 5: Allocate send and recv buf");
  size_t buf_size = 0;
  if (nrecv != 0) {
    recvbuf = new BufArray1D<Real>[nrecv];
    for (int n : nmyrblist) {
      int on = newtoold[n];
      LogicalLocation &oloc = loclist[on];
      LogicalLocation &nloc = newloc[n];
      if (oloc.level > nloc.level) { // f2c
        for (int l = 0; l < nleaf; l++) {
          if (ranklist[on + l] == Globals::my_rank) continue;
          buf_size += bsf2c;
        }
      } else { // same level or c2f
        if (ranklist[on] == Globals::my_rank) continue;
        int size;
        if (oloc.level == nloc.level) {
          size = bssame;
        } else {
          size = bsc2f;
        }
        buf_size += size;
      }
    }
  }
  if (nsend != 0) {
    sendbuf = new BufArray1D<Real>[nsend];
    for (int n : omyrblist) {
      int nn = oldtonew[n];
      LogicalLocation &oloc = loclist[n];
      LogicalLocation &nloc = newloc[nn];
      auto pb = FindMeshBlock(n);
      if (nloc.level == oloc.level) { // same level
        if (newrank[nn] == Globals::my_rank) continue;
        buf_size += bssame;
      } else if (nloc.level > oloc.level) { // c2f
        // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
        for (int l = 0; l < nleaf; l++) {
          if (newrank[nn + l] == Globals::my_rank) continue;
          buf_size += bsc2f;
        } // end loop over nleaf (unique to c2f branch in this step 6)
      } else { // f2c: restrict + pack + send
        if (newrank[nn] == Globals::my_rank) continue;
        buf_size += bsf2c;
      }
    }
  }
  BufArray1D<Real> bufs("RedistributeAndRefineMeshBlocks sendrecv bufs", buf_size);
  Kokkos::Profiling::popRegion(); // Step 5

  // Step 6. allocate and start receiving buffers
  Kokkos::Profiling::pushRegion("Step 6: Pack buffer and start recv");
  size_t buf_offset = 0;
  if (nrecv != 0) {
    req_recv = new MPI_Request[nrecv];
    int rb_idx = 0; // recv buffer index
    for (int nidx = 0; nidx < nmyrblist.size(); nidx++) {
      int n = nmyrblist[nidx];
      int on = newtoold[n];
      LogicalLocation &oloc = loclist[on];
      LogicalLocation &nloc = newloc[n];
      if (oloc.level > nloc.level) { // f2c
        for (int l = 0; l < nleaf; l++) {
          if (ranklist[on + l] == Globals::my_rank) continue;
          LogicalLocation &lloc = loclist[on + l];
          int ox1 = ((lloc.lx1 & 1LL) == 1LL), ox2 = ((lloc.lx2 & 1LL) == 1LL),
              ox3 = ((lloc.lx3 & 1LL) == 1LL);
          recvbuf[rb_idx] =
              BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bsf2c));
          buf_offset += bsf2c;
          int tag = CreateAMRMPITag(nidx, ox1, ox2, ox3);
          PARTHENON_MPI_CHECK(MPI_Irecv(recvbuf[rb_idx].data(), bsf2c, MPI_PARTHENON_REAL,
                                        ranklist[on + l], tag, MPI_COMM_WORLD,
                                        &(req_recv[rb_idx])));
          rb_idx++;
        }
      } else { // same level or c2f
        if (ranklist[on] == Globals::my_rank) continue;
        int size;
        if (oloc.level == nloc.level) {
          size = bssame;
        } else {
          size = bsc2f;
        }
        recvbuf[rb_idx] =
            BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + size));
        buf_offset += size;
        int tag = CreateAMRMPITag(nidx, 0, 0, 0);
        PARTHENON_MPI_CHECK(MPI_Irecv(recvbuf[rb_idx].data(), size, MPI_PARTHENON_REAL,
                                      ranklist[on], tag, MPI_COMM_WORLD,
                                      &(req_recv[rb_idx])));
        rb_idx++;
      }
    }
  }
  Kokkos::Profiling::popRegion(); // Step 6

  // Step 7. allocate, pack and start sending buffers
  Kokkos::Profiling::pushRegion("Step 7: Pack and send buffers");
  if (nsend != 0) {
    req_send = new MPI_Request[nsend];
    std::vector<int> tags(nsend);
    std::vector<int> dest(nsend);
    std::vector<int> count(nsend);
    int sb_idx = 0; // send buffer index
    for (int nidx = 0; nidx < omyrblist.size(); nidx++) {
      int n = omyrblist[nidx];
      int nn = oldtonew[n];
      LogicalLocation &oloc = loclist[n];
      LogicalLocation &nloc = newloc[nn];
      auto pb = FindMeshBlock(n);
      if (nloc.level == oloc.level) { // same level
        if (newrank[nn] == Globals::my_rank) continue;
        sendbuf[sb_idx] =
            BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bssame));
        buf_offset += bssame;
        PrepareSendSameLevel(pb.get(), sendbuf[sb_idx]);
        int nn_lid = GetLid(rblist, newrank, nn);
        tags[sb_idx] = CreateAMRMPITag(nn_lid, 0, 0, 0);
        dest[sb_idx] = newrank[nn];
        count[sb_idx] = bssame;
        sb_idx++;
      } else if (nloc.level > oloc.level) { // c2f
        // c2f must communicate to multiple leaf blocks (unlike f2c, same2same)
        for (int l = 0; l < nleaf; l++) {
          if (newrank[nn + l] == Globals::my_rank) continue;
          sendbuf[sb_idx] =
              BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bsc2f));
          buf_offset += bsc2f;
          PrepareSendCoarseToFineAMR(pb.get(), sendbuf[sb_idx], newloc[nn + l]);
          int nnl_lid = GetLid(rblist, newrank, nn + l);
          tags[sb_idx] = CreateAMRMPITag(nnl_lid, 0, 0, 0);
          dest[sb_idx] = newrank[nn + l];
          count[sb_idx] = bsc2f;
          sb_idx++;
        } // end loop over nleaf (unique to c2f branch in this step 6)
      } else { // f2c: restrict + pack + send
        if (newrank[nn] == Globals::my_rank) continue;
        sendbuf[sb_idx] =
            BufArray1D<Real>(bufs, std::make_pair(buf_offset, buf_offset + bsf2c));
        buf_offset += bsf2c;
        PrepareSendFineToCoarseAMR(pb.get(), sendbuf[sb_idx]);
        int ox1 = ((oloc.lx1 & 1LL) == 1LL), ox2 = ((oloc.lx2 & 1LL) == 1LL),
            ox3 = ((oloc.lx3 & 1LL) == 1LL);
        int nn_lid = GetLid(rblist, newrank, nn);
        tags[sb_idx] = CreateAMRMPITag(nn_lid, ox1, ox2, ox3);
        dest[sb_idx] = newrank[nn];
        count[sb_idx] = bsf2c;
        sb_idx++;
      }
    }
    // wait until all send buffers are filled
    Kokkos::fence();
    for (auto idx = 0; idx < sb_idx; idx++) {
      PARTHENON_MPI_CHECK(MPI_Isend(sendbuf[idx].data(), count[idx], MPI_PARTHENON_REAL,
                                    dest[idx], tags[idx], MPI_COMM_WORLD,
                                    &(req_send[idx])));
    }
  } // if (nsend !=0)
  Kokkos::Profiling::popRegion(); // Step 7
#endif                            // MPI_PARALLEL

  // Step 8. construct a new MeshBlock list (moving the data within the MPI rank)
  Kokkos::Profiling::pushRegion("Step 8: Construct new MeshBlockList");
  {
    RegionSize block_size = GetBlockSize();

    BlockList_t new_block_list(nmyrblist.size());
    for (int nidx = 0; nidx < nmyrblist.size(); nidx++) {
      int n = nmyrblist[nidx];
      int on = newtoold[n];
      if ((ranklist[on] == Globals::my_rank) && (loclist[on].level == newloc[n].level)) {
        // on the same MPI rank and same level -> just move it
        new_block_list[nidx] = FindMeshBlock(on);
      } else {
        // on a different refinement level or MPI rank - create a new block
        BoundaryFlag block_bcs[6];
        SetBlockSizeAndBoundaries(newloc[n], block_size, block_bcs);
        // append new block to list of MeshBlocks
        new_block_list[nidx] =
            MeshBlock::Make(n, nidx, newloc[n], block_size, block_bcs, this, pin, app_in,
                            packages, resolved_packages, gflag);
        // fill the conservative variables
        if ((loclist[on].level > newloc[n].level)) { // fine to coarse (f2c)
          for (int ll = 0; ll < nleaf; ll++) {
            if (ranklist[on + ll] != Globals::my_rank) continue;
            // fine to coarse on the same MPI rank (different AMR level) - restriction
            auto pob = FindMeshBlock(on + ll);

            // allocate sparse variables that were allocated on old block
            for (auto var : pob->meshblock_data.Get()->GetVariableVector()) {
              if (var->IsSparse() && var->IsAllocated()) {
                new_block_list[nidx]->AllocateSparse(var->label());
              }
            }
            FillSameRankFineToCoarseAMR(pob.get(), new_block_list[nidx].get(),
                                        loclist[on + ll]);
          }
        } else if ((loclist[on].level < newloc[n].level) && // coarse to fine (c2f)
                   (ranklist[on] == Globals::my_rank)) {
          // coarse to fine on the same MPI rank (different AMR level) - prolongation
          auto pob = FindMeshBlock(on);

          // allocate sparse variables that were allocated on old block
          for (auto var : pob->meshblock_data.Get()->GetVariableVector()) {
            if (var->IsSparse() && var->IsAllocated()) {
              new_block_list[nidx]->AllocateSparse(var->label());
            }
          }
          FillSameRankCoarseToFineAMR(pob.get(), new_block_list[nidx].get(), newloc[n]);
        }
      }
    }

    // Replace the MeshBlock list
    block_list = std::move(new_block_list);
    gid_lid_map.clear();

    // Ensure local and global ids are correct
    for (int nidx = 0; nidx < nmyrblist.size(); nidx++) {
      int n = nmyrblist[nidx];
      block_list[nidx]->gid = n;
      block_list[nidx]->lid = nidx;
      gid_lid_map[n] = nidx;
    }
  }
  Kokkos::Profiling::popRegion(); // Step 8: Construct new MeshBlockList

  // Step 9. Receive the data and load into MeshBlocks
  Kokkos::Profiling::pushRegion("Step 9: Recv data and unpack");
  // This is a test: try MPI_Waitall later.
#ifdef MPI_PARALLEL
  if (nrecv != 0) {
    int test;
    std::vector<bool> received(nrecv, false);
    int rb_idx;
    do {
      rb_idx = 0; // recv buffer index
      for (int nidx = 0; nidx < nmyrblist.size(); nidx++) {
        int n = nmyrblist[nidx];
        int on = newtoold[n];
        LogicalLocation &oloc = loclist[on];
        LogicalLocation &nloc = newloc[n];
        auto pb = FindMeshBlock(n);
        if (oloc.level == nloc.level) { // same
          if (ranklist[on] == Globals::my_rank) continue;
          if (!received[rb_idx]) {
            PARTHENON_MPI_CHECK(MPI_Test(&(req_recv[rb_idx]), &test, MPI_STATUS_IGNORE));
            if (static_cast<bool>(test)) {
              FinishRecvSameLevel(pb.get(), recvbuf[rb_idx]);
              received[rb_idx] = true;
            }
          }
          rb_idx++;
        } else if (oloc.level > nloc.level) { // f2c
          for (int l = 0; l < nleaf; l++) {
            if (ranklist[on + l] == Globals::my_rank) continue;
            if (!received[rb_idx]) {
              PARTHENON_MPI_CHECK(
                  MPI_Test(&(req_recv[rb_idx]), &test, MPI_STATUS_IGNORE));
              if (static_cast<bool>(test)) {
                FinishRecvFineToCoarseAMR(pb.get(), recvbuf[rb_idx], loclist[on + l]);
                received[rb_idx] = true;
              }
            }
            rb_idx++;
          }
        } else { // c2f
          if (ranklist[on] == Globals::my_rank) continue;
          if (!received[rb_idx]) {
            PARTHENON_MPI_CHECK(MPI_Test(&(req_recv[rb_idx]), &test, MPI_STATUS_IGNORE));
            if (static_cast<bool>(test)) {
              FinishRecvCoarseToFineAMR(pb.get(), recvbuf[rb_idx]);
              received[rb_idx] = true;
            }
          }
          rb_idx++;
        }
      }
      // rb_idx is a running index, so we repeat the loop until all vals are true
    } while (!std::all_of(received.begin(), received.begin() + rb_idx,
                          [](bool v) { return v; }));
    Kokkos::fence();
  }
#endif

  // deallocate arrays
  newtoold.clear();
  oldtonew.clear();
#ifdef MPI_PARALLEL
  if (nsend != 0) {
    PARTHENON_MPI_CHECK(MPI_Waitall(nsend, req_send, MPI_STATUSES_IGNORE));
    delete[] sendbuf;
    delete[] req_send;
  }
  if (nrecv != 0) {
    delete[] recvbuf;
    delete[] req_recv;
  }
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  Kokkos::Profiling::popRegion(); // Step 9

  // update the lists
  loclist = std::move(newloc);
  ranklist = std::move(newrank);
  costlist = std::move(newcost);

  // re-initialize the MeshBlocks
  for (auto &pmb : block_list) {
    pmb->pbval->SearchAndSetNeighbors(tree, rblist, ranklist);
  }
  Initialize(false, pin, app_in);

  ResetLoadBalanceVariables();

  MPI_Barrier(MPI_COMM_WORLD);
  Kokkos::Profiling::popRegion(); // RedistributeAndRefineMeshBlocks
                                  //
  auto _rnr_time = tau::GetUsSince(_ts_beg);
  tau::LogBlockEvent(-1, TAU_BLKEVT_US_COMP3, _rnr_time);

  perf.Pause();
}

// AMR: step 6, branch 1 (same2same: just pack+send)

void Mesh::PrepareSendSameLevel(MeshBlock *pmb, BufArray1D<Real> &sendbuf) {
  // inital offset, data starts after allocation flags
  int p = pmb->vars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(sendbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);
  // this helper fn is used for AMR and non-refinement load balancing of
  // MeshBlocks. Therefore, unlike PrepareSendCoarseToFineAMR(), etc., it loops over
  // MeshBlock::vars_cc/fc_ containers, not MeshRefinement::pvars_cc/fc_ containers

  // TODO(felker): add explicit check to ensure that elements of pmb->vars_cc/fc_ and
  // pmb->pmr->pvars_cc/fc_ v point to the same objects, if adaptive

  // (C++11) range-based for loop: (automatic type deduction fails when iterating over
  // container with std::reference_wrapper; could use auto var_cc_r = var_cc.get())
  for (int i = 0; i < pmb->vars_cc_.size(); ++i) {
    auto &pvar_cc = pmb->vars_cc_[i];
    alloc_subview_h(i) = pvar_cc->IsAllocated() ? 1.0 : 0.0;
    int nu = pvar_cc->GetDim(4) - 1;
    if (pvar_cc->IsAllocated()) {
      ParArray4D<Real> var_cc = pvar_cc->data.Get<4>();
      BufferUtility::PackData(var_cc, sendbuf, 0, nu, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e,
                              p, pmb);
    } else {
      // increment offset
      p += (nu + 1) * (ib.e + 1 - ib.s) * (jb.e + 1 - jb.s) * (kb.e + 1 - kb.s);
    }
  }

  Kokkos::deep_copy(alloc_subview, alloc_subview_h);

  // WARNING(felker): casting from "Real *" to "int *" in order to append single integer
  // to send buffer is slightly unsafe (especially if sizeof(int) > sizeof(Real))
  if (adaptive) {
    Kokkos::deep_copy(pmb->exec_space,
                      Kokkos::View<int, Kokkos::MemoryUnmanaged>(
                          reinterpret_cast<int *>(Kokkos::subview(sendbuf, p).data())),
                      pmb->pmr->deref_count_);
  }
  return;
}

// step 6, branch 2 (c2f: just pack+send)

void Mesh::PrepareSendCoarseToFineAMR(MeshBlock *pb, BufArray1D<Real> &sendbuf,
                                      LogicalLocation &lloc) {
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d
  int ox1 = static_cast<int>((lloc.lx1 & 1LL) == 1LL);
  int ox2 = static_cast<int>((lloc.lx2 & 1LL) == 1LL);
  int ox3 = static_cast<int>((lloc.lx3 & 1LL) == 1LL);
  const IndexDomain interior = IndexDomain::interior;
  // pack
  int il, iu, jl, ju, kl, ku;
  if (ox1 == 0) {
    il = pb->cellbounds.is(interior) - 1;
    iu = pb->cellbounds.is(interior) + pb->block_size.nx1 / 2;
  } else {
    il = pb->cellbounds.is(interior) + pb->block_size.nx1 / 2 - 1;
    iu = pb->cellbounds.ie(interior) + 1;
  }
  if (ox2 == 0) {
    jl = pb->cellbounds.js(interior) - f2;
    ju = pb->cellbounds.js(interior) + pb->block_size.nx2 / 2;
  } else {
    jl = pb->cellbounds.js(interior) + pb->block_size.nx2 / 2 - f2;
    ju = pb->cellbounds.je(interior) + f2;
  }
  if (ox3 == 0) {
    kl = pb->cellbounds.ks(interior) - f3;
    ku = pb->cellbounds.ks(interior) + pb->block_size.nx3 / 2;
  } else {
    kl = pb->cellbounds.ks(interior) + pb->block_size.nx3 / 2 - f3;
    ku = pb->cellbounds.ke(interior) + f3;
  }

  // inital offset, data starts after allocation flags
  int p = pb->pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(sendbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  int i = 0;
  for (auto &cc_var : pb->pmr->pvars_cc_) {
    alloc_subview_h(i) = cc_var->IsAllocated() ? 1.0 : 0.0;
    int nu = cc_var->GetDim(4) - 1; // TODO(JMM): looks like this only supports vectors?
    if (cc_var->IsAllocated()) {
      ParArray4D<Real> var_cc = cc_var->data.Get<4>();
      BufferUtility::PackData(var_cc, sendbuf, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
    } else {
      BufferUtility::PackZero(sendbuf, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
    }
    i++;
  }

  Kokkos::deep_copy(alloc_subview, alloc_subview_h);

  return;
}

// step 6, branch 3 (f2c: restrict, pack, send)

void Mesh::PrepareSendFineToCoarseAMR(MeshBlock *pb, BufArray1D<Real> &sendbuf) {
  // restrict and pack

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pb->c_cellbounds.GetBoundsK(interior);

  auto &pmr = pb->pmr;

  // inital offset, data starts after allocation flags
  int p = pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(sendbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  int i = 0;
  for (auto &cc_var : pmr->pvars_cc_) {
    alloc_subview_h(i) = cc_var->IsAllocated() ? 1.0 : 0.0;
    int nu = cc_var->GetDim(4) - 1;
    if (cc_var->IsAllocated()) {
      pmr->RestrictCellCenteredValues(cc_var.get(), cib.s, cib.e, cjb.s, cjb.e, ckb.s,
                                      ckb.e);
      // TOGO(pgrete) remove temp var once Restrict func interface is updated
      ParArray4D<Real> coarse_cc = (cc_var->coarse_s).Get<4>();
      BufferUtility::PackData(coarse_cc, sendbuf, 0, nu, cib.s, cib.e, cjb.s, cjb.e,
                              ckb.s, ckb.e, p, pb);
    } else {
      BufferUtility::PackZero(sendbuf, 0, nu, cib.s, cib.e, cjb.s, cjb.e, ckb.s, ckb.e, p,
                              pb);
    }
    i++;
  }

  Kokkos::deep_copy(alloc_subview, alloc_subview_h);

  return;
}

// step 7: f2c, same MPI rank, different level (just restrict+copy, no pack/send)

void Mesh::FillSameRankFineToCoarseAMR(MeshBlock *pob, MeshBlock *pmb,
                                       LogicalLocation &loc) {
  auto &pmr = pob->pmr;
  const IndexDomain interior = IndexDomain::interior;
  int il =
      pmb->cellbounds.is(interior) + ((loc.lx1 & 1LL) == 1LL) * pmb->block_size.nx1 / 2;
  int jl =
      pmb->cellbounds.js(interior) + ((loc.lx2 & 1LL) == 1LL) * pmb->block_size.nx2 / 2;
  int kl =
      pmb->cellbounds.ks(interior) + ((loc.lx3 & 1LL) == 1LL) * pmb->block_size.nx3 / 2;

  IndexRange cib = pob->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pob->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pob->c_cellbounds.GetBoundsK(interior);
  // absent a zip() feature for range-based for loops, manually advance the
  // iterator over "SMR/AMR-enrolled" cell-centered quantities on the new
  // MeshBlock in lock-step with pob
  auto pmb_cc_it = pmb->pmr->pvars_cc_.begin();
  // iterate MeshRefinement std::vectors on pob
  for (auto cc_var : pmr->pvars_cc_) {
    const bool fine_allocated = cc_var->IsAllocated();
    if (!(*pmb_cc_it)->IsAllocated()) {
      PARTHENON_REQUIRE_THROWS(!fine_allocated,
                               "Mesh::FillSameRankFineToCoarseAMR: Destination not "
                               "allocated but source allocated");
      pmb_cc_it++;
      continue;
    }
    if (fine_allocated) {
      pmr->RestrictCellCenteredValues(cc_var.get(), cib.s, cib.e, cjb.s, cjb.e, ckb.s,
                                      ckb.e);
    }

    // copy from old/original/other MeshBlock (pob) to newly created block (pmb)
    ParArrayND<Real> src = cc_var->coarse_s;
    ParArrayND<Real> dst = (*pmb_cc_it)->data;
    int nu = cc_var->GetDim(4) - 1;
    int koff = kl - ckb.s;
    int joff = jl - cjb.s;
    int ioff = il - cib.s;
    pmb->par_for(
        "FillSameRankFineToCoarseAMR", 0, nu, ckb.s, ckb.e, cjb.s, cjb.e, cib.s, cib.e,
        KOKKOS_LAMBDA(const int nv, const int k, const int j, const int i) {
          // if the destination (coarse) is allocated, but source (fine) is not allocated,
          // we just fill destination with 0's
          dst(nv, k + koff, j + joff, i + ioff) = fine_allocated ? src(nv, k, j, i) : 0.0;
        });
    pmb_cc_it++;
  }

  return;
}

// step 7: c2f, same MPI rank, different level (just copy+prolongate, no pack/send)

void Mesh::FillSameRankCoarseToFineAMR(MeshBlock *pob, MeshBlock *pmb,
                                       LogicalLocation &newloc) {
  auto &pmr = pmb->pmr;

  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  int il = pob->c_cellbounds.is(interior) - 1;
  int iu = pob->c_cellbounds.ie(interior) + 1;
  int jl = pob->c_cellbounds.js(interior) - f2;
  int ju = pob->c_cellbounds.je(interior) + f2;
  int kl = pob->c_cellbounds.ks(interior) - f3;
  int ku = pob->c_cellbounds.ke(interior) + f3;

  int cis = ((newloc.lx1 & 1LL) == 1LL) * pob->block_size.nx1 / 2 +
            pob->cellbounds.is(interior) - 1;
  int cjs = ((newloc.lx2 & 1LL) == 1LL) * pob->block_size.nx2 / 2 +
            pob->cellbounds.js(interior) - f2;
  int cks = ((newloc.lx3 & 1LL) == 1LL) * pob->block_size.nx3 / 2 +
            pob->cellbounds.ks(interior) - f3;

  auto pob_cc_it = pob->pmr->pvars_cc_.begin();
  // iterate MeshRefinement std::vectors on new pmb
  for (auto cc_var : pmr->pvars_cc_) {
    PARTHENON_REQUIRE_THROWS(cc_var->IsAllocated() == (*pob_cc_it)->IsAllocated(),
                             "Mesh::FillSameRankCoarseToFineAMR: Allocation mismatch");
    if (!cc_var->IsAllocated()) {
      pob_cc_it++;
      continue;
    }

    int nu = cc_var->GetDim(4) - 1;
    ParArrayND<Real> src = (*pob_cc_it)->data;
    ParArrayND<Real> dst = cc_var->coarse_s;
    // fill the coarse buffer
    // WARNING: potential Cuda stream pitfall (exec space of coarse and fine MB)
    // Need to make sure that both src and dst are done with all other task up to here
    pob->par_for(
        "FillSameRankCoarseToFineAMR", 0, nu, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int nv, const int k, const int j, const int i) {
          dst(nv, k, j, i) = src(nv, k - kl + cks, j - jl + cjs, i - il + cis);
        });
    // keeping the original, following block for reference to indexing
    // for (int nv = 0; nv <= nu; nv++) {
    //   for (int k = kl, ck = cks; k <= ku; k++, ck++) {
    //     for (int j = jl, cj = cjs; j <= ju; j++, cj++) {
    //       for (int i = il, ci = cis; i <= iu; i++, ci++)
    //         dst(nv, k, j, i) = src(nv, ck, cj, ci);
    //     }
    //   }
    // }
    pmr->ProlongateCellCenteredValues(
        cc_var.get(), pob->c_cellbounds.is(interior), pob->c_cellbounds.ie(interior),
        pob->c_cellbounds.js(interior), pob->c_cellbounds.je(interior),
        pob->c_cellbounds.ks(interior), pob->c_cellbounds.ke(interior));
    pob_cc_it++;
  }
  return;
}

// step 8 (receive and load), branch 1 (same2same: unpack)
void Mesh::FinishRecvSameLevel(MeshBlock *pmb, BufArray1D<Real> &recvbuf) {
  // inital offset, data starts after allocation flags
  int p = pmb->vars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(recvbuf, std::make_pair(0, p));
  auto alloc_subview_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), alloc_subview);

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pmb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(interior);

  for (int i = 0; i < pmb->vars_cc_.size(); ++i) {
    auto &pvar_cc = pmb->vars_cc_[i];
    int nu = pvar_cc->GetDim(4) - 1;

    if (alloc_subview_h(i) == 1.0) {
      // allocated on sending block
      if (!pvar_cc->IsAllocated()) {
        // need to allocate locally
        pmb->AllocateSparse(pvar_cc->label());
      }
      PARTHENON_REQUIRE_THROWS(
          pvar_cc->IsAllocated(),
          "FinishRecvSameLevel: Received variable that was allocated on sending "
          "block but it is not allocated on receiving block");
      ParArray4D<Real> var_cc_ = pvar_cc->data.Get<4>();
      BufferUtility::UnpackData(recvbuf, var_cc_, 0, nu, ib.s, ib.e, jb.s, jb.e, kb.s,
                                kb.e, p, pmb);
    } else {
      // increment offset
      p += (nu + 1) * (ib.e + 1 - ib.s) * (jb.e + 1 - jb.s) * (kb.e + 1 - kb.s);
      PARTHENON_REQUIRE_THROWS(
          !pvar_cc->IsAllocated(),
          "FinishRecvSameLevel: Received variable that was not allocated on sending "
          "block but it is allocated on receiving block");
    }
  }

  // WARNING(felker): casting from "Real *" to "int *" in order to read single
  // appended integer from received buffer is slightly unsafe
  if (adaptive) {
    Kokkos::deep_copy(pmb->exec_space, pmb->pmr->deref_count_,
                      Kokkos::View<int, Kokkos::MemoryUnmanaged>(
                          reinterpret_cast<int *>(Kokkos::subview(recvbuf, p).data())));
  }
  return;
}

// step 8 (receive and load), branch 2 (f2c: unpack)
void Mesh::FinishRecvFineToCoarseAMR(MeshBlock *pb, BufArray1D<Real> &recvbuf,
                                     LogicalLocation &lloc) {
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d

  const IndexDomain interior = IndexDomain::interior;
  IndexRange ib = pb->cellbounds.GetBoundsI(interior);
  IndexRange jb = pb->cellbounds.GetBoundsJ(interior);
  IndexRange kb = pb->cellbounds.GetBoundsK(interior);

  int ox1 = static_cast<int>((lloc.lx1 & 1LL) == 1LL);
  int ox2 = static_cast<int>((lloc.lx2 & 1LL) == 1LL);
  int ox3 = static_cast<int>((lloc.lx3 & 1LL) == 1LL);
  int il, iu, jl, ju, kl, ku;

  if (ox1 == 0)
    il = ib.s, iu = ib.s + pb->block_size.nx1 / 2 - 1;
  else
    il = ib.s + pb->block_size.nx1 / 2, iu = ib.e;
  if (ox2 == 0)
    jl = jb.s, ju = jb.s + pb->block_size.nx2 / 2 - f2;
  else
    jl = jb.s + pb->block_size.nx2 / 2, ju = jb.e;
  if (ox3 == 0)
    kl = kb.s, ku = kb.s + pb->block_size.nx3 / 2 - f3;
  else
    kl = kb.s + pb->block_size.nx3 / 2, ku = kb.e;

  // inital offset, data starts after allocation flags
  int p = pb->pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(recvbuf, std::make_pair(0, p));
  auto alloc_subview_h =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), alloc_subview);

  int i = 0;
  for (auto &cc_var : pb->pmr->pvars_cc_) {
    int nu = cc_var->GetDim(4) - 1;

    if ((alloc_subview_h(i) == 1.0) && !cc_var->IsAllocated()) {
      // need to allocate locally
      pb->AllocateSparse(cc_var->label());
      PARTHENON_REQUIRE_THROWS(
          cc_var->IsAllocated(),
          "Mesh::FinishRecvFineToCoarseAMR: Failed to allocate variable");
    }

    if (cc_var->IsAllocated()) {
      ParArray4D<Real> var_cc = cc_var->data.Get<4>();
      BufferUtility::UnpackData(recvbuf, var_cc, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
    } else {
      // increment offset
      p += (nu + 1) * (iu + 1 - il) * (ju + 1 - jl) * (ku + 1 - kl);
    }
    i++;
  }
  return;
}

// step 8 (receive and load), branch 2 (c2f: unpack+prolongate)
void Mesh::FinishRecvCoarseToFineAMR(MeshBlock *pb, BufArray1D<Real> &recvbuf) {
  const int f2 = (ndim >= 2) ? 1 : 0; // extra cells/faces from being 2d
  const int f3 = (ndim >= 3) ? 1 : 0; // extra cells/faces from being 3d
  auto &pmr = pb->pmr;
  // inital offset, data starts after allocation flags
  int p = pmr->pvars_cc_.size();

  // subview to set allocation flags
  auto alloc_subview = Kokkos::subview(recvbuf, std::make_pair(0, p));
  auto alloc_subview_h = Kokkos::create_mirror_view(HostMemSpace(), alloc_subview);

  const IndexDomain interior = IndexDomain::interior;
  IndexRange cib = pb->c_cellbounds.GetBoundsI(interior);
  IndexRange cjb = pb->c_cellbounds.GetBoundsJ(interior);
  IndexRange ckb = pb->c_cellbounds.GetBoundsK(interior);

  int il = cib.s - 1, iu = cib.e + 1, jl = cjb.s - f2, ju = cjb.e + f2, kl = ckb.s - f3,
      ku = ckb.e + f3;

  int i = 0;
  for (auto &cc_var : pmr->pvars_cc_) {
    int nu = cc_var->GetDim(4) - 1;

    if ((alloc_subview_h(i) == 1.0) && !cc_var->IsAllocated()) {
      // need to allocate locally
      pb->AllocateSparse(cc_var->label());
      PARTHENON_REQUIRE_THROWS(
          cc_var->IsAllocated(),
          "Mesh::FinishRecvCoarseToFineAMR: Failed to allocate variable");
    }

    if (cc_var->IsAllocated()) {
      PARTHENON_REQUIRE_THROWS(nu == cc_var->GetDim(4) - 1, "nu mismatch");
      ParArray4D<Real> coarse_cc = (cc_var->coarse_s).Get<4>();
      BufferUtility::UnpackData(recvbuf, coarse_cc, 0, nu, il, iu, jl, ju, kl, ku, p, pb);
      pmr->ProlongateCellCenteredValues(cc_var.get(), cib.s, cib.e, cjb.s, cjb.e, ckb.s,
                                        ckb.e);
    } else {
      // increment offset
      p += (nu + 1) * (iu + 1 - il) * (ju + 1 - jl) * (ku + 1 - kl);
    }
    i++;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3)
//  \brief calculate an MPI tag for AMR block transfer
// tag = local id of destination (remaining bits) + ox1(1 bit) + ox2(1 bit) + ox3(1 bit)
//       + physics(5 bits)

// See comments on BoundaryBase::CreateBvalsMPITag()

int Mesh::CreateAMRMPITag(int lid, int ox1, int ox2, int ox3) {
  // the trailing zero is used as "id" to indicate an AMR related tag
  return (lid << 8) | (ox1 << 7) | (ox2 << 6) | (ox3 << 5) | 0;
}

} // namespace parthenon
