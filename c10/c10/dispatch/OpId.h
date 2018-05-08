#pragma once

#include <c10/guts/IdWrapper.h>

namespace c10 {

/**
 * Unique identifier for an operator signature, e.g., identifies 'add' versus 'mul' distinctly.
 *
 * NB: Every distinct overload of an operator, e.g., mm(Tensor, Tensor)
 * versus mm(double alpha, Tensor, double beta, Tensor), gets a DISTINCT
 * OpId.
 */
class OpId final : public guts::IdWrapper<OpId, uint32_t> {
public:
  // TODO Don't allow public constructor
  constexpr explicit OpId(uint32_t id): IdWrapper(id) {}
};

}

C10_DEFINE_HASH_FOR_IDWRAPPER(c10::OpId)
