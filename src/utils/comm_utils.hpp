#include <iostream>
#include <sstream>
#include <memory>

namespace parthenon {
//             Read    Write
//    stale:             X
// sending*:
// received:     X
enum class BufferState { stale, sending, sending_null, received, received_null };

enum class BuffCommType { sender, receiver, both, sparse_receiver };

class CommBufferLogging {
  public:
  static const char* BufState2Str(BufferState const& state) {
    switch (state) {
    case BufferState::stale:
      return "stale";
    case BufferState::sending:
      return "sending";
    case BufferState::sending_null:
      return "sending_null";
    case BufferState::received:
      return "received";
    case BufferState::received_null:
      return "received_null";
    default:
      return "unknown";
    }
  }

  static const char* BufCommType2Str(BuffCommType const& type) {
    switch (type) {
    case BuffCommType::sender:
      return "sender";
    case BuffCommType::receiver:
      return "receiver";
    case BuffCommType::both:
      return "both";
    case BuffCommType::sparse_receiver:
      return "sparse_receiver";
    default:
      return "unknown";
    }
  }

  static void LogThingsInCommBuffer(
      std::shared_ptr<BufferState> state,
      std::shared_ptr<BuffCommType> comm_type,
      int size,
      std::shared_ptr<bool> started_irecv,
      int my_rank,
      int tag,
      int send_rank,
      int recv_rank,
      bool active
      ) {
    std::stringstream ss;
    ss << "BufferState: " << BufState2Str(*state) << ", "
       << "BuffCommType: " << BufCommType2Str(*comm_type) << ", "
       << "Bufsize: " << size << ", "
       << "started_irecv: " << *started_irecv << ", "
       << "my_rank: " << my_rank << ", "
       << "tag: " << tag << ", "
       << "send_rank: " << send_rank << ", "
       << "recv_rank: " << recv_rank << ", "
       << "active: " << active;

    std::cout << "CommBufferState: " << ss.str() << std::endl;
  }

  static uint64_t GetMsSince(uint64_t ms_start) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    uint64_t ms_now = ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    return ms_now - ms_start;
  }
};
};
