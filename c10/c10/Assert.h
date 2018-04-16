#pragma once

// assert used to test for internal errors
#define C10_ASSERT(e, ...) (void)(e);

// check for user errors in use of API
// TODO: build this out
#define C10_CHECK(e, ...) if (!(e)) { throw std::runtime_error("user test failed"); }
