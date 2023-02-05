#include "../common.h"

#define CEIL_TO_POWER_OF_2(x) (((x) <= 2 ? 2 : ((x) <= 4 ? 4 : ((x) <= 8 ? 8 : ((x) <= 16 ? 16 : ((x) <= 32 ? 32 : \
                              ((x) <= 64 ? 64 : ((x) <= 128 ? 128 : ((x) <= 256 ? 256 : ((x) <= 512 ? 512 : 1024))))))))))

// Inclusive warp scan
template <uint num_lanes = WARP_SIZE, class T>
__device__ static void inc_warp_scan(// output
	                                 T &prefix_sum, T &total_sum, 
	                                 // input
	                                 T value, int warp_lane) {
	const uint effec_num_lanes = CEIL_TO_POWER_OF_2(num_lanes);
	const uint shfl_mask = ~((~0) << effec_num_lanes);
	
	if (effec_num_lanes != num_lanes) {
		prefix_sum = (warp_lane < num_lanes ? value : 0);
	}	
	else {
		prefix_sum = value;
	}
	for (int delta = 1; delta < effec_num_lanes; delta <<= 1) {
		uint temp = __shfl_up_sync(shfl_mask, prefix_sum, delta, effec_num_lanes);
		if (warp_lane >= delta) prefix_sum += temp;
	}
	
	total_sum = __shfl_sync(shfl_mask, prefix_sum, effec_num_lanes - 1);
}

// Exclusive warp scan
template <uint num_lanes = WARP_SIZE, class T>
__device__ static void exc_warp_scan(// output
	                                 T &prefix_sum, T &total_sum,
	                                 // input
	                                 T value, int warp_lane) {
	inc_warp_scan<num_lanes, T>(prefix_sum, total_sum, value, warp_lane);
	prefix_sum = prefix_sum - value;
}
