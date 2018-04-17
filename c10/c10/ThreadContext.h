#pragma once

namespace c10 {

// The thread context is a per-thread context which is NOT EXPECTED to be preserved
// across threads; that is, it is used to apply *truly* thread-local settings, e.g.,
// as might be set by RAII guards.  There is a global default thread context, which is
// copied in to initialize the thread context when it is not initialized.
class ThreadContext final {

};

ThreadContext &threadContext();
ThreadContext &defaultThreadContext();

}
