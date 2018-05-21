#ifndef CAFFE2_CORE_TYPEID_H_
#define CAFFE2_CORE_TYPEID_H_

#include "../../c10/c10/guts/TypeId.h"

// Thin wrapper for c10::TypeId/TypeMeta.
// TODO Remove this file and use c10 directly

namespace caffe2 {

using CaffeTypeId = c10::TypeId;

using TypeMeta = c10::TypeMeta;

// Needs to be called from ::caffe2 namespace
#define CAFFE_KNOWN_TYPE(T)                              \
  } namespace c10 {                                      \
  using namespace ::caffe2;                              \
  C10_KNOWN_TYPE(T)                                      \
  } namespace caffe2 {

} // namespace caffe2

#endif // CAFFE2_CORE_TYPEID_H_
