#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

#include <tuple>

using namespace std;

namespace gpu {
	const int kBlockPerGrid = 8192;
	const int kThreadPerBlock = 512;
	const int kWarpSize = 32;
	const unsigned kFullMask = 0xFFFFFFFF;

	template<class T>
	__device__ T WarpReduce(T value) {
	#pragma unroll
	    for (int delta = 1; delta < kWarpSize; delta *= 2)
	#if __CUDACC_VER_MAJOR__ >= 9
	        value += __shfl_down_sync(kFullMask, value, delta);
	#else
	        value += __shfl_down(value, delta);
	#endif
	    return value;
	}

	template<class T>
	__device__ T WarpBroadcast(T value, int lane_id) {
	#if __CUDACC_VER_MAJOR__ >= 9
	    return __shfl_sync(kFullMask, value, lane_id);
	#else
	    return __shfl(value, lane_id);
	#endif
	}
}

namespace cppext_rotate_compare {
	using namespace at;

	#define INIT_VARS \
		int64_t thi = blockIdx.x * blockDim.x + threadIdx.x; \
		int64_t thn = gridDim.x * blockDim.x;

	template<class scalar_t> __global__ 
	void forward_out(int64_t n, int64_t d,
		const scalar_t *a, const scalar_t *b, 
		const int64_t* pa, const int64_t* pb,
		scalar_t *dist) {

		INIT_VARS
		thn >>= 5;

		for(int64_t pos = (thi >> 5); pos < n; pos += thn) {
			const scalar_t *are = a + (2 * d * pa[pos]);
			const scalar_t *bre = b + (2 * d * pb[pos]);
			const scalar_t *aim = are + d, *bim = bre + d;
			scalar_t value = 0, re, im;
			for(int64_t i = (thi & 31); i < d; i += 32) {
				re = are[i] - bre[i];
				im = aim[i] - bim[i];
				value += sqrtf(re * re + im * im);
			}
			value = gpu::WarpBroadcast(gpu::WarpReduce(value), 0);
			if((thi & 31) == 0) dist[pos] = value;
		}

	}

	template<class scalar_t> __global__ 
	void backward_out(int64_t n, int64_t d,
		const scalar_t *a, const scalar_t *b, 
		const int64_t* pa, const int64_t* pb,
		const scalar_t *ogdist,
		scalar_t* iga, scalar_t* igb) {

		INIT_VARS
		thn >>= 5;

		for(int64_t pos = (thi >> 5); pos < n; pos += thn) {
			int64_t pap = pa[pos], pbp = pb[pos];
			const scalar_t *are = a + (2 * d * pap);
			const scalar_t *bre = b + (2 * d * pbp);
			const scalar_t *aim = are + d, *bim = bre + d;
			scalar_t *igare = iga + (2 * d * pap);
			scalar_t *igbre = igb + (2 * d * pbp);
			scalar_t *igaim = igare + d, *igbim = igbre + d;
			scalar_t tmp, re, im, ogdist_pos = ogdist[pos];

			for(int64_t i = (thi & 31); i < d; i += 32) {
				re = are[i] - bre[i];
				im = aim[i] - bim[i];
				tmp = ogdist_pos / fmaxf(sqrtf(re * re + im * im), 1e-6);
				re *= tmp;
				im *= tmp;

				atomicAdd(igare + i, re);
				atomicAdd(igaim + i, im);
				atomicAdd(igbre + i, -re);
				atomicAdd(igbim + i, -im);
			}
		}
	}

	Tensor forward(const Tensor& a, const Tensor& b, const Tensor& pa, const Tensor& pb) {
		TORCH_CHECK(a.dim() == 2, "a should be 2D");
		TORCH_CHECK(b.dim() == 2, "b should be 2D");
		TORCH_CHECK(pa.dim() == 1, "pa should be 1D");
		TORCH_CHECK(pb.dim() == 1, "pb should be 1D");

		int64_t n = pa.size(0), d = a.size(1) / 2;

		TORCH_CHECK(pb.size(0) == n, "pa and pb should have same size");
		TORCH_CHECK(a.size(1) == d * 2, "a should have size [n, 2d]");
		TORCH_CHECK(b.size(1) == d * 2, "b should have size [n, 2d]");

		Tensor dist = zeros({n}, a.options());

		AT_DISPATCH_FLOATING_TYPES(a.type(), "forward_out", [&] {
			forward_out<scalar_t><<<512, 512>>>(
				n, d,
				a.data_ptr<scalar_t>(),
				b.data_ptr<scalar_t>(),
				pa.data_ptr<int64_t>(),
				pb.data_ptr<int64_t>(),
				dist.data_ptr<scalar_t>()
			);
		});

		return dist;
	}

	tuple<Tensor, Tensor, Tensor, Tensor>
	backward(const Tensor& a, const Tensor& b, const Tensor& pa, const Tensor& pb, const Tensor& ogdist) {
		TORCH_CHECK(a.dim() == 2, "a should be 2D");
		TORCH_CHECK(b.dim() == 2, "b should be 2D");
		TORCH_CHECK(pa.dim() == 1, "pa should be 1D");
		TORCH_CHECK(pb.dim() == 1, "pb should be 1D");

		int64_t n = pa.size(0), d = a.size(1) / 2;

		TORCH_CHECK(pb.size(0) == n, "pa and pb should have same size");
		TORCH_CHECK(a.size(1) == d * 2, "a should have size [n, 2d]");
		TORCH_CHECK(b.size(1) == d * 2, "b should have size [n, 2d]");

		Tensor iga = zeros(a.sizes().vec(), a.options());
		Tensor igb = zeros(b.sizes().vec(), b.options());
		Tensor igp = zeros({1}, a.options());

		AT_DISPATCH_FLOATING_TYPES(a.type(), "backward_out", [&] {
			backward_out<scalar_t><<<512, 512>>>(
				n, d,
				a.data_ptr<scalar_t>(),
				b.data_ptr<scalar_t>(),
				pa.data_ptr<int64_t>(),
				pb.data_ptr<int64_t>(),
				ogdist.data_ptr<scalar_t>(),
				iga.data_ptr<scalar_t>(),
				igb.data_ptr<scalar_t>()
			);
		});

		return make_tuple(iga, igb, igp, igp);
	}

}

