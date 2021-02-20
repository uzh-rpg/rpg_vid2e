import torch
import esim_cuda


class EventSimulator_torch(torch.nn.Module):
    def __init__(self, contrast_threshold):
        self.contrast_threshold = contrast_threshold
        self.initial_reference_values = None
        self.last_image = None
        self.last_time = None

    def _check_inputs(self, images, timestamps):
        assert timestamps.dtype == torch.int64, timestamps.dtype
        assert images.dtype == torch.float32, images.dtype

    def reset(self):
        self.initial_reference_values = None
        self.last_image = None
        self.last_time = None

    def forward(self,
                images, 
                timestamps):

        if len(images.shape) == 2:
            images = images.unsqueeze(0)
        if len(timestamps.shape) == 0:
            timestamps = timestamps.unsqueeze(0)

        self._check_inputs(images, timestamps)

        if self.initial_reference_values is None:
            self.initial_reference_values = images[0].clone()

        if self.last_image is not None:
            images = torch.cat([self.last_image, images], 0)
            timestamps = torch.cat([self.last_time, timestamps], 0)

        if len(images) == 1:
            self.last_image = images[-1:]
            self.last_time = timestamps[-1:]
            return None

        events = self.initialized_forward(images, timestamps)

        self.last_image = images[-1:]
        self.last_time = timestamps[-1:]

        return events

    def initialized_forward(self, images, timestamps):

        T, H, W = images.shape
        reference_values_over_time = torch.zeros((T-1, H, W),
                                                 device=images.device,
                                                 dtype=images.dtype)

        event_counts = torch.zeros_like(images[0]).long()

        reference_values_over_time, event_counts = esim_cuda.forward_count_events(images, 
                                                                                  self.initial_reference_values,
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
                                   self.initial_reference_values,
                                   reference_values_over_time,
                                   offsets,
                                   events,
                                   self.contrast_threshold)

        # sort by timestamps. Do this for each batch of events
        events = events[events[:,2].argsort()]

        self.initial_reference_values = reference_values_over_time[-1]


        return dict(zip(['x','y','t','p'], events.T))