
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> esim_forward(
    const torch::Tensor& images,
    const torch::Tensor& timestamps,
    const torch::Tensor& init_reference_values,
    const torch::Tensor& reference_values_over_time,
    const torch::Tensor& offsets,
    torch::Tensor& events,
    torch::Tensor& timestamps_last_event,
    float contrast_threshold_negative,
    float contrast_threshold_positive,
    int64_t refractory_period);

std::vector<torch::Tensor> esim_forward_count_events(
    const torch::Tensor& images,
    const torch::Tensor& timestamps,
    const torch::Tensor& init_reference_values,
    torch::Tensor& reference_values_over_time
    torch::Tensor& event_counts,
    torch::Tensor& timestamps_last_event,
    float contrast_threshold_negative,
    float contrast_threshold_positive);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &esim_forward, "ESIM forward (CUDA)");
  m.def("forward_count_events", &esim_forward_count_events, "ESIM forward count events (CUDA)");
}