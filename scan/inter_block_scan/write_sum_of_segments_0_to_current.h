#include "sum_field_macros.h"

__device__ static void write_sum_of_segments_0_to_current(// output
                                                          volatile ull *d_scan,
                                                          // input            
                                                          int segment_idx,
                                                          ull sum_of_segments_0_to_current) {
	d_scan[segment_idx * WARP_SIZE + WARP_SIZE - 1] = SET_FLAG_P(sum_of_segments_0_to_current);
}
