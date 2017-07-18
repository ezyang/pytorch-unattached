#include "torch/csrc/jit/assert.h"

#include <cstdarg>
#include <cstdio>

namespace torch { namespace jit {

void
barf(const char *fmt, ...)
{
  char msg[2048];
  va_list args;

  va_start(args, fmt);
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  // If you uncomment this, gdb will break at this point so you
  // can inspect the backtrace.  DO NOT commit this, since this
  // will cause a hard failure rather than a soft one.
  // __builtin_trap();

  throw assert_error(msg);
}

}}
