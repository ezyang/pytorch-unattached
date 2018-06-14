#pragma once

// On Windows, you must explicitly specify what symbols will become publically available
// from your DLL using __declspec(dllexport); otherwise, they will be hidden.
//
// If you get a linker error on Windows but not on Linux, this is probably because you forgot
// to tag a non-inlined function as C10_API.
//
// It is possible to force shared library creation on Windows to export all symbols, but DLLs have
// a symbol limit, so this is generally not a good idea.  (However, C10 is intended to never have
// too many symbols, so this should be unlikely to be a problem.)

#ifdef _WIN32
# ifdef C10_EXPORTS
#  define C10_API __declspec(dllexport)
# else
#  define C10_API __declspec(dllimport)
# endif
#else
# define C10_API
#endif