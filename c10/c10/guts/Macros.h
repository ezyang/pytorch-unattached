#pragma once

/**
 * Disallow the copy and assignment constructors of a class
 */
#define DISALLOW_COPY_AND_ASSIGN(Class)         \
  Class(const Class &rhs) = delete;             \
  Class &operator=(const Class &rhs) = delete;  \
