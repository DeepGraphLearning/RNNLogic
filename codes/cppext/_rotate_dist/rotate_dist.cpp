#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

#include <tuple>

using namespace std;

namespace cppext_rotate_dist {
	using namespace at;

	Tensor forward(const Tensor& x, const Tensor& a);

	tuple<Tensor, Tensor>
	backward(const Tensor& x, const Tensor& a, const Tensor& outgrad_dist);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", cppext_rotate_dist::forward, "forward");
    m.def("backward", cppext_rotate_dist::backward, "backward");
}