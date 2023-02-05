#include "../../common.h"

// For the last element of each d_scan segment (i.e., d_scan[31], d_scan[63], d_scan[95]...),
// indicates whether it contains the corresponding inclusive prefix sum.
#define FLAG_P 0x8000000000000000

#define SET_FLAG_P(sum) ((sum) | FLAG_P)
#define FLAG_P_IS_SET(sum) ((sum) & FLAG_P)

// The sum field is stored in the first 7 bits of each d_scan element (i.e., bits 63 to 57).
// It is composed of the following sub-fields:
// - Bit 63: flag P.
// - Bits 62 to 57: sums counter.
#define NUM_OF_BITS_OF_SUM_FIELD 7

#define NUM_OF_BITS_OF_PREFIX_SUM (64 - NUM_OF_BITS_OF_SUM_FIELD)

#define INIT_SUM_FIELD(value) \
		(1LLU << NUM_OF_BITS_OF_PREFIX_SUM) | ((ull)(value))

#define GET_SUM_FIELD(value) ((uint)((value) >> NUM_OF_BITS_OF_PREFIX_SUM))

#define ALL_SUMS_PERFORMED(num_of_sums_in_segment_lane, segment_lane) \
		                  ((num_of_sums_in_segment_lane) == ((segment_lane) + 1))		

#define SUM_FIELD_MASK  (((1LLU << NUM_OF_BITS_OF_SUM_FIELD) - 1) << NUM_OF_BITS_OF_PREFIX_SUM)
#define DEL_SUM_FIELD(value) ((value) & ~SUM_FIELD_MASK)

