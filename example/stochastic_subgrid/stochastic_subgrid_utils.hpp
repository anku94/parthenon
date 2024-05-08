#pragma once

#include "mesh/meshblock_pack.hpp"

namespace parthenon {
void SetupBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();

  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto v = rc->Get("num_iter").data;

  pmb->par_for(
      "stochastic_subgrid_package::DoLotsOfWork", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) { v(k, j, i) = 50; });
}
} // namespace parthenon
