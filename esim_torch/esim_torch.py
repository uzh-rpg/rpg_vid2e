import torch
import esim_cuda


class EventSimulator_torch(torch.nn.Module):
    def __init__(self, contrast_threshold):
        self.contrast_threshold = contrast_threshold

    def _check_inputs(self, images, timestamps):
        assert timestamps.dtype == torch.int64, timestamps.dtype
        assert images.dtype == torch.float32, images.dtype

    def forward(self,
                images, 
                timestamps):

        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        if len(timestamps.shape) == 0:
            timestamps = timestamps.unsqueeze(0)

        self._check_inputs(images, timestamps)

        initial_reference_values = images[0].clone()

        if len(images) == 0:
            return None

        events = self.initialized_forward(images, timestamps,
                                          initial_reference_values)

        return events

    def initialized_forward(self, images, timestamps,
                            initial_reference_values):

        T, H, W = images.shape
        reference_values_over_time = torch.zeros((T, H, W),
                                                 device=images.device,
                                                 dtype=images.dtype)

        event_counts = torch.zeros_like(images[0]).long()

        reference_values_over_time, event_counts = esim_cuda.forward_count_events(images, 
                                                                                initial_reference_values,
                                                                                reference_values_over_time,
                                                                                event_counts, 
                                                                                self.contrast_threshold)

        # compute the offsets for each event group
        cumsum = event_counts.view(-1).cumsum(dim=0)
        total_num_events = cumsum[-1]
        offsets = cumsum.view(H, W) - event_counts

        # compute events on the GPU
        events = torch.zeros((total_num_events, 4), device=cumsum.device, dtype=cumsum.dtype)

        events = esim_cuda.forward(images,
                                    timestamps,
                                    initial_reference_values,
                                    reference_values_over_time,
                                    offsets,
                                    events,
                                    self.contrast_threshold)

        # sort by timestamps. Do this for each batch of events
        events = events[events[:,2].argsort()]

        return dict(zip(['x','y','t','p'], events.T))