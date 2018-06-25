#include <memory>
#include "ThreadContext.h"

namespace c10 {

ThreadContext &threadContext() {
  static thread_local std::unique_ptr<ThreadContext> thread_context;
  if (!thread_context) thread_context = std::unique_ptr<ThreadContext>(new ThreadContext(defaultThreadContext()));
  return *thread_context;
}

ThreadContext &defaultThreadContext() {
  static ThreadContext default_thread_context;
  return default_thread_context;
}

}