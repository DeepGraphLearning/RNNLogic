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

namespace cppext_rotate_dist {
	using namespace at;

	#define INIT_VARS \
		int64_t thi = blockIdx.x * blockDim.x + threadIdx.x; \
		int64_t thn = gridDim.x * blockDim.x;

	template<class scalar_t>
	__global__ void forward_out(const int64_t n, const int64_t d, 
		const scalar_t *x, const scalar_t *a, scalar_t* dist) {

		INIT_VARS
		thn >>= 5;

		for(int64_t pos = (thi >> 5); pos < n; pos += thn) {
			const scalar_t *a_pos = a + pos * 2 * d;
			scalar_t re, im, value = 0;
			for(int64_t i = (thi & 31); i < d; i += 32) {
				re = x[i] - a_pos[i];
				im = x[i + d] - a_pos[i + d];
				value += sqrtf(re * re + im * im);
			}
			value = gpu::WarpBroadcast(gpu::WarpReduce(value), 0);
			if((thi & 31) == 0) dist[pos] = value;
		}
	}

	template<class scalar_t>
	__global__ void backward_out(const int64_t n, const int64_t d, 
		const scalar_t *x, const scalar_t *a, const scalar_t *outgrad_dist,
		scalar_t *ingrad_x, scalar_t *ingrad_a) {

		INIT_VARS
		thn >>= 5;

///*bgg*/if(thi==0) for(int i=0;i<n;++i) printf("%d = %.4lf\n",i, outgrad_dist[i]);

		const scalar_t *xre = x, *xim = x + d;
		scalar_t *gxre = ingrad_x, *gxim = ingrad_x + d;

		for(int64_t i = (thi >> 5); i < n; i += thn) {
			scalar_t re, im, tmp;
			const scalar_t o_i = outgrad_dist[i];
			const scalar_t *are_i = a + (i * 2 * d);
			const scalar_t *aim_i = a + (i * 2 * d) + d;
			scalar_t *gare_i = ingrad_a + (i * 2 * d);
			scalar_t *gaim_i = ingrad_a + (i * 2 * d) + d;


			for(int64_t j = (thi & 31); j < d; j += 32) {
				re = xre[j] - are_i[j];
				im = xim[j] - aim_i[j];
///*bgg*/scalar_t dis=sqrtf(re * re + im * im);
				tmp = o_i / sqrtf(re * re + im * im);
///*bgg*/printf("%lld %lld %.4f %.4f\n", i,j,dis,o_i);
				re *= tmp;
				im *= tmp;
				atomicAdd(gxre + j, re);
				atomicAdd(gxim + j, im);
				gare_i[j] = -re;
				gaim_i[j] = -im; 
			}
		}
	}

	Tensor forward(const Tensor& x, const Tensor& a) {
		TORCH_CHECK(x.dim() == 1, "x should be 1D");
		TORCH_CHECK(a.dim() == 2, "a should be 2D");
		int64_t n = a.size(0), d = x.size(0) / 2;
		TORCH_CHECK(x.size(0) == d * 2, "x should have size [2d]");
		TORCH_CHECK(a.size(1) == d * 2, "a should have size [n, 2d] with same d");

		Tensor dist = zeros({n}, x.options());

		AT_DISPATCH_FLOATING_TYPES(x.type(), "forward_out", [&] {
			forward_out<scalar_t><<<512, 512>>>(
				n, d,
				x.data_ptr<scalar_t>(),
				a.data_ptr<scalar_t>(),

				dist.data_ptr<scalar_t>()
			);
		});

		return dist;
	}

	tuple<Tensor, Tensor>
	backward(const Tensor& x, const Tensor& a, const Tensor& outgrad_dist) {
		TORCH_CHECK(x.dim() == 1, "x should be 1D");
		TORCH_CHECK(a.dim() == 2, "a should be 2D");
		int64_t n = a.size(0), d = x.size(0) / 2;
		// printf("read n = %lld d = %lld x (%lld) a (%lld, %lld)\n", 
		// 	n,d,x.size(0),a.size(0),a.size(1));
		TORCH_CHECK(x.size(0) == d * 2, "x should have size [2d]");
		TORCH_CHECK(a.size(1) == d * 2, "a should have size [n, 2d] with same d");

///*bgg*/auto aaa=outgrad_dist.data_ptr<float>();
///*bgg*/for(int i=0;i<n;++i) printf("%d = %.4f\n",i, aaa[i]);

		Tensor ingrad_x = zeros({2*d}, x.options());
		Tensor ingrad_a = zeros({n, 2*d}, x.options());

		AT_DISPATCH_FLOATING_TYPES(x.type(), "backward_out", [&] {
			backward_out<scalar_t><<<512, 512>>>(
				n, d,
				x.data_ptr<scalar_t>(),
				a.data_ptr<scalar_t>(),
				outgrad_dist.data_ptr<scalar_t>(),

				ingrad_x.data_ptr<scalar_t>(),
				ingrad_a.data_ptr<scalar_t>()
			);
		});
///*bgg*/cudaDeviceSynchronize();
		return make_tuple(ingrad_x, ingrad_a);

	}
	
}