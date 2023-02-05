#ifndef _COMMON_H
#define _COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define WARP_SIZE 32
#define WARP_IDX(thread_idx) ((thread_idx) >> 5)
#define WARP_LANE(thread_idx) ((thread_idx) & (WARP_SIZE - 1))
#define NUM_WARPS(block_size) ((block_size) / WARP_SIZE)

#define uchar unsigned char
#define ushort unsigned short
#define uint unsigned int
#define ull unsigned long long
struct __align__(32) uchar32 {
	uchar symbol[32];
};

#endif // _COMMON_H