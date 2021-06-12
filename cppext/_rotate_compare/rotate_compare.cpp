#include <torch/extension.h>
#include <ATen/ParallelOpenMP.h>

#include <tuple>

using namespace std;

namespace cppext_rotate_compare {
	using namespace at;

	Tensor forward(const Tensor& a, const Tensor& b, const Tensor& pa, const Tensor& pb);

	tuple<Tensor, Tensor, Tensor, Tensor>
	backward(const Tensor& a, const Tensor& b, const Tensor& pa, const Tensor& pb, const Tensor& outgrad_dist);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", cppext_rotate_compare::forward, "forward");
    m.def("backward", cppext_rotate_compare::backward, "backward");
}