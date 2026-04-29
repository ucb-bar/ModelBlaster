/*
 * spin_inject.h — force-injected at the top of every pthreadpool TU
 * via gcc's -include flag (only for the BENCH_PTHREADPOOL_VARIANT=spin
 * build). We pull in pthreadpool's own threadpool-common.h FIRST,
 * which defines PTHREADPOOL_SPIN_WAIT_ITERATIONS to 1_000_000, then
 * #undef + #redefine it to our spin-friendly 100_000_000. Because the
 * upstream header is `#pragma once`, its later inclusion from
 * src/pthreads.c (etc.) is a no-op and our value sticks.
 *
 * This file lives in spin_overrides/ but does NOT need a matching
 * filename — `-include` takes the file path verbatim. We name it
 * spin_inject.h to avoid the include-name collision that the earlier
 * approach (a clone of threadpool-common.h) ran into:
 * `#include "threadpool-common.h"` from src/pthreads.c looks up the
 * header relative to src/ first, so our copy was never picked.
 */

#pragma once

/* This works because src/threadpool-common.h is on pthreadpool's
 * private include path (target_include_directories(pthreadpool PRIVATE
 * src)), and our `-include` runs in the same compile context. */
#include "threadpool-common.h"

#undef PTHREADPOOL_SPIN_WAIT_ITERATIONS
#define PTHREADPOOL_SPIN_WAIT_ITERATIONS 100000000
