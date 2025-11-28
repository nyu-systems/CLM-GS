# Overview of `--clm_offload`

Please read the **Common Setup** section in the main README.md first to understand the common implementations in all three strategies.

This is our highest-performance CPU offloading strategy, implementing the design described in our paper. It can train the largest models and minimizes offloading overheads for both communication and CPU-based Adam updates. However, the implementation is complex; we recommend understanding `--naive_offload` before learning `--clm_offload`, as this provides a smoother learning curve. Additionally, please also read our paper first to understand how the design maps to the code implementation. 

## Gaussian Model (`./gaussian_model.py:GaussianModelCLMOffload`)

We store `xyz`, `scaling`, `rotation`, and `opacity` on GPU, totaling 11 32-bit floating-point numbers per Gaussian. In contrast, `_features_dc` and `_features_rest` are stored in CPU pinned memory, totaling 48 32-bit floating-point numbers per Gaussian. 

## Rendering Engine (`./engine.py`)

The `clm_offload_train_one_batch()` function is the core implementation for training on a batch of camera views. The pipeline works as follows:

1. **Compute visibility**: The `calculate_filters()` function computes the indices of in-frustum Gaussians for every camera in the batch, stored in `filters`. This is critical because subsequent rendering operations only use these visible Gaussians' parameters. This step does not require CPU-to-GPU communication because all relevant attributes are natively stored on GPU.

2. **Compute camera rendering order**: Before rendering starts, CLM computes the optimal camera ordering to maximize speed by calling `order_calculation()`. See Section 4.2.3 (Pipeline Order Optimization) in our paper for the algorithm behind this implementation. The function contains numerous bitwise operations, some of which are implemented in the `clm_kernels` submodule, making the code somewhat complex. Currently, `order_calculation()` only supports batch sizes of 4, 8, 16, 32, and 64, which limits `--clm_offload`'s available batch size choices.

   After determining the order, the function also computes the indices of Gaussians at each camera rendering step that can begin Adam updates (i.e., those that will not be used in subsequent rendering). These indices are stored in `finish_indices_filters`. Additionally, it computes the buffer sizes for communicating Gaussians between CPU and GPU at each step (`cnt_h`, `cnt_d`, and `cnt_g`). These pre-compuated buffer sizes are used to allocate buffers without Synchronizing the whole GPU. 

3. **Initialize CPU Adam thread**: We launch a concurrent CPU thread to perform Adam updates in the background. This thread continuously updates Gaussians that have gradients ready in the current batch. We declare `signal_tensor_pinned` in CPU pinned memory to indicate whether the i-th camera in the batch has finished its rendering, backward pass, and gradient transfer back to CPU. Once `signal_tensor_pinned[i]` becomes true, Gaussians in `finish_indices_filters[i]` can perform Adam updates. The GPU communication stream sets `signal_tensor_pinned[i]` to true once the kernel for sending gradients back to pinned memory completes. The `parameters_grad_buffer` receives and accumulates gradients from GPU.

4. **Render and backpropagate over all cameras**: We loop over each camera in the batch. Figure 6 in our paper illustrates how the pipeline works at the algorithm level. We have two GPU streams here: a communication stream and a computation stream. Rendering kernels and their backward kernels are launched to the computation stream.

Kernels used to load parameters to GPU (`send_shs2gpu_stream()`, `send_shs2gpu_stream_retention()`) and offload gradients to CPU (`send_shs2cpu_grad_buffer_stream_retention()`, `send_shs2cpu_grad_buffer_stream()`) are launched to the communication stream. We implement custom CUDA kernels for these four communication functions. Because accessed Gaussian indices are always at random positions in memory, we use CUDA DMA in the CUDA kenerls to avoid extra data copies on either CPU or GPU. See `submodules/clm_kernel/` for our kernel implementations.

We use CUDA events to synchronize between communication and computation streams. The `cpu2gpu_event` ensures that parameter loading on the communication stream finishes before rendering on the computation stream, while `gpu2cpu_event` ensures that rendering backward completes before calling kernels that send gradients back to CPU.

There is another synchronization point between the communication stream and the CPU Adam thread. Specifically, after the kernel for sending gradients back to CPU finishes on the communication stream for the i-th camera, `clm_kernels.set_signal(...)` sets `signal_tensor_pinned[i]` to true. This operation uses DMA from GPU to CPU pinned memory, with `__threadfence_system()` added to ensure gradients are fully written back. The CPU Adam thread uses a busy-wait loop to wait for `signal_tensor_pinned[i]` to become true before performing Adam updates.

Additionally, we use `.scatter_add_()` and `.gather()` to accelerate sparse operations throughout the code.

The rendering operations themselves follow standard procedures without special modifications.

5. **GPU Adam update**: At the end of batch training, we update the parameters residing on GPU. 
