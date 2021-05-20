#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

/*
Precompute the reference values and number of events between the reference values to read out easily after
*/
template <typename scalar_t>
__global__ void count_events_cuda_forward_kernel(
    const scalar_t* __restrict__ imgs,
    const scalar_t* __restrict__ init_refs,
    scalar_t* __restrict__ refs_over_time,
    int64_t* __restrict__ count_ev, 
    int T, int H, int W, float ct_neg, float ct_pos)
{
  // linear index
  const int linIdx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // check that thread is not out of valid range
  if (linIdx >= H * W)
    return;

  scalar_t ref = init_refs[linIdx];
  int tot_num_events = 0;
  for (int t=0; t<T-1; t++)
  {
    int tidx = (t+1) * H * W + linIdx;
    int tidx_min_1 = t * H * W + linIdx;

    scalar_t i0 = imgs[tidx_min_1];
    scalar_t i1 = imgs[tidx];

    int num_events, polarity; 

    // process events leading up to i1. 
    polarity = (i1 >= ref) ? 1 : -1;
    float ct = (i1 >= ref) ? ct_pos : ct_neg;
    num_events = std::abs(i1 - ref) / ct;
    tot_num_events += num_events;
    ref += polarity * ct * num_events;

    // store number of events and reference values for later
    // triggered_events_t stores the number of events between t-1 and t
    // refs_t stores the reference at t.
    refs_over_time[tidx_min_1] = ref;
  }
  count_ev[linIdx] = tot_num_events;
}
    

template <typename scalar_t>
__global__ void esim_cuda_forward_kernel(
  const scalar_t* __restrict__ imgs,
  const int64_t* __restrict__ ts,
  const scalar_t* __restrict__ init_ref,
  const scalar_t* __restrict__ refs_over_time,
  const int64_t* __restrict__ offsets,
  int64_t* __restrict__ ev,
  int64_t* __restrict__ t_last_ev,
  int T, int H, int W, float ct_neg, float ct_pos, int64_t t_ref
) 
{
  // linear index
  const int linIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // check that thread is not out of valid range
  if (linIdx >= H * W)
    return;

  int x = linIdx % W;
  int y = linIdx / W;

  scalar_t ref0 = init_ref[linIdx];
  int64_t offset = offsets[linIdx];

  for (int t=0; t<T-1; t++) {
  
    // offset_t stores the offset at t.

    scalar_t i0 = imgs[linIdx+(t)*H*W]; // shifts forward one timestamp 
    scalar_t i1 = imgs[linIdx+(t+1)*H*W]; // shifts forward one timestamp 

    int64_t t0 = ts[t];
    int64_t t1 = ts[t+1];
    
    if (t > 0) {
      ref0 = refs_over_time[linIdx+(t-1)*H*W];
    }

    int polarity = (i1 >= ref0) ? 1 : -1;
    float ct = (i1 >= ref0) ? ct_pos : ct_neg;
    int64_t num_events = std::abs(i1 - ref0) / ct;

    int64_t t_prev = t_last_ev[linIdx];
    for (int evIdx=0; evIdx<num_events; evIdx++) 
    {
      scalar_t r = (ref0 + (evIdx+1) * polarity * ct - i0) / (i1 - i0);
      int64_t timestamp = t0 + (t1-t0)*r;
      int64_t delta_t = timestamp - t_prev;


      if (delta_t > t_ref || t_prev == 0) {
          int64_t idx = 4 * (offset + evIdx);
          ev[idx + 0] = x;
          ev[idx + 1] = y;
          ev[idx + 2] = timestamp;
          ev[idx + 3] = polarity;
          t_last_ev[linIdx] = timestamp;
          t_prev = timestamp;
      }
    } 
    offset += num_events;
  }
}

std::vector<torch::Tensor> esim_forward_count_events(
  const torch::Tensor& imgs,       // T x H x W
  const torch::Tensor& init_refs,  // H x W
  torch::Tensor& refs_over_time,   // T-1 x H x W
  torch::Tensor& count_ev,         // H x W
  float ct_neg,
  float ct_pos)
{
  CHECK_INPUT(imgs);
  CHECK_INPUT(count_ev);
  CHECK_INPUT(init_refs);
  CHECK_INPUT(refs_over_time);
  CHECK_DEVICE(imgs, count_ev);
  CHECK_DEVICE(imgs, init_refs);
  CHECK_DEVICE(imgs, refs_over_time);

  //cudaSetDevice(imgs.device().index());
  
  unsigned T = imgs.size(0);
  unsigned H = imgs.size(1);
  unsigned W = imgs.size(2);
  
  //unsigned MAX_NUM_EVENTS = ev.size(1);
  
  unsigned threads = 256;
  dim3 blocks((H * W + threads - 1) / threads, 1);

  count_events_cuda_forward_kernel<float><<<blocks, threads>>>(
      imgs.data<float>(), 
      init_refs.data<float>(),
      refs_over_time.data<float>(),
      count_ev.data<int64_t>(),
      T, H, W, ct_neg, ct_pos
    );

  return {refs_over_time, count_ev};
}

torch::Tensor esim_forward(
    const torch::Tensor& imgs, // T x H x W
    const torch::Tensor& ts, // T
    const torch::Tensor& init_refs, // H x W
    const torch::Tensor& refs_over_time, // T-1 x H x W
    const torch::Tensor& offsets, // H x W 
    torch::Tensor& ev,  // N x 4, x y t p
    torch::Tensor& t_last_ev,  // H x W
    float ct_neg,
    float ct_pos,
    int64_t dt_ref
  ) 
{
  CHECK_INPUT(imgs);
  CHECK_INPUT(ts);
  CHECK_INPUT(ev);
  CHECK_INPUT(offsets);
  CHECK_INPUT(refs_over_time);
  CHECK_INPUT(init_refs);
  
  CHECK_DEVICE(imgs, ts);
  CHECK_DEVICE(imgs, ev);
  CHECK_DEVICE(imgs, offsets);
  CHECK_DEVICE(imgs, init_refs);
  CHECK_DEVICE(imgs, refs_over_time);
  CHECK_DEVICE(imgs, t_last_ev);

  //cudaSetDevice(imgs.device().index());

  unsigned T = imgs.size(0);
  unsigned H = imgs.size(1);
  unsigned W = imgs.size(2);

  unsigned threads = 256;
  dim3 blocks((H * W + threads - 1) / threads, 1);

  esim_cuda_forward_kernel<float><<<blocks, threads>>>(
      imgs.data<float>(),
      ts.data<int64_t>(), 
      init_refs.data<float>(),
      refs_over_time.data<float>(),
      offsets.data<int64_t>(),
      ev.data<int64_t>(),
      t_last_ev.data<int64_t>(),
      T, H, W, ct_neg, ct_pos, dt_ref
    );
  
  return ev;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &esim_forward, "ESIM forward (CUDA)");
  m.def("forward_count_events", &esim_forward_count_events, "ESIM forward count events (CUDA)");
}