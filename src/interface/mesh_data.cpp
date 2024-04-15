#include "interface/mesh_data.hpp"

#include "mesh/mesh.hpp"

namespace parthenon {
template <>
void MeshData<double>::AddBlockCost(int block_idx, double cost) {
  // block_idx is relative to this meshdata obj
  int block_lid = block_lids_[block_idx];
  pmy_mesh_->block_list[block_lid]->AddCostForLoadBalancing(cost);
}
} // namespace parthenon
