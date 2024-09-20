//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022. Triad National Security, LLC. All rights reserved.
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

#ifndef UTILS_COMMUNICATION_BUFFER_HPP_
#define UTILS_COMMUNICATION_BUFFER_HPP_

#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>

#include "globals.hpp"
#include "outputs/tau_types.hpp"
#include "parthenon_mpi.hpp"
#include "utils/comm_utils.hpp"
#include "utils/drain_queue.hpp"
#include "utils/error_checking.hpp"
#include "utils/mpi_types.hpp"

namespace {}

namespace parthenon {

template <class T>
class CommBuffer {
 private:
  // Need specializations to be friends with each other
  template <typename U>
  friend class CommBuffer;

  std::shared_ptr<BufferState> state_;
  std::shared_ptr<BuffCommType> comm_type_;
  std::shared_ptr<bool> started_irecv_;
  std::shared_ptr<int> nrecv_tries_;
  std::shared_ptr<mpi_request_t> my_request_;

  int my_rank;
  int tag_;
  int send_rank_;
  int recv_rank_;
  mpi_comm_t comm_;

  using buf_base_t = std::remove_pointer_t<decltype(std::declval<T>().data())>;
  buf_base_t null_buf_ = std::numeric_limits<buf_base_t>::signaling_NaN();
  bool active_ = false;

  std::function<T()> get_resource_;

  T buf_;

 public:
  CommBuffer()
      : my_rank(0)
#ifdef MPI_PARALLEL
        ,
        my_request_(std::make_shared<MPI_Request>(MPI_REQUEST_NULL))
#endif
  {
  }

  CommBuffer(int tag, int send_rank, int recv_rank, mpi_comm_t comm_,
             std::function<T()> get_resource, bool do_sparse_allocation = false);

  ~CommBuffer();

  template <class U>
  CommBuffer(const CommBuffer<U> &in);

  template <class U>
  CommBuffer &operator=(const CommBuffer<U> &in);

  operator T &() { return buf_; }
  operator const T &() const { return buf_; }

  T &buffer() { return buf_; }
  const T &buffer() const { return buf_; }

  void Allocate() {
    if (!active_) {
      buf_ = get_resource_();
      active_ = true;
    }
  }

  void Free() {
    buf_ = T();
    active_ = false;
  }

  bool IsActive() const { return active_; }

  BufferState GetState() { return *state_; }

  void Send() noexcept;
  void SendNull() noexcept;
  void DrainPendingSends(SendDrainQueue &dq);

  bool IsSendPending();
  std::shared_ptr<MPI_Request> ReleaseMPIReq();

  bool IsAvailableForWrite();

  void TryStartReceive() noexcept;
  bool TryReceive() noexcept;

  void Stale();

  void SendGap();
};
} // namespace parthenon
#endif // UTILS_COMMUNICATION_BUFFER_HPP_
