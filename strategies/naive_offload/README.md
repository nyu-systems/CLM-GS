# Overview of `--naive_offload`

Please read the **Common Setup** section in the main README.md first to understand the common implementations in all three strategies.

This is a simple CPU offloading implementation that leverages CPU memory to support larger 3D Gaussian Splatting model training than `--no_offload`. This implementation is easy to understand and implement, with no complex logic; however, it makes training significantly slower. We recommend understanding `--naive_offload` before learning `--clm_offload`, as it provides a smoother learning curve. 

The strategy works as follows: it stores all Gaussian attributes (xyz, etc.) and their optimizer states on CPU, loads parameters onto GPU in each iteration, performs the forward and backward computation, offloads gradients back to CPU in each batch, and performs Adam optimizer updates on the CPU. This is essentially a [Zero-offload](https://www.deepspeed.ai/tutorials/zero-offload/)-style approach applied to 3DGS, adapted from the original design for neural network offloading. 

## Gaussian Model (`./gaussian_model.py:GaussianModelNaiveOffload`)

The only difference between the Gaussian model in `--naive_offload` and the original 3D Gaussian Splatting model from [gaussian-splatting/scene/gaussian_model.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py) is that all states (parameters and corresponding optimizer states) in `GaussianModelNaiveOffload` are stored on CPU instead of GPU.

**Pinned memory for faster communication:** The `.pin_memory()` call in `./gaussian_model.py:GaussianModelNaiveOffload.create_from_pcd()` speeds up data communication between CPU and GPU. Pinned memory is essential to fully utilize the PCIe hardware bandwidth. 

## Rendering Engine (`./engine.py`)

The `naive_offload_train_one_batch()` function is the core implementation for training on a batch of camera views. The pipeline works as follows:

1. **Load parameters to GPU**: Before rendering, all Gaussian parameters are loaded from CPU to GPU in a single transfer.

2. **Compute visibility**: The `calculate_filters()` function computes the indices of in-frustum Gaussians for every camera in the batch. This is critical because subsequent rendering operations only use these visible Gaussians' parameters.

3. **Initialize gradient buffers**: Gradient buffers are initialized on GPU to accumulate gradients across training cameras. Another gradient buffer in CPU pinned memory is initialized to receive gradients from the GPU buffer after all cameras are rendered.

4. **Render and backpropagate**: For each camera in the batch, `torch.gather()` uses the in-frustum indices (from step 2) to extract only the visible Gaussian parameters. The same GPU rendering process as in `--no_offload` is performed, followed by backpropagation. Gradients are accumulated across cameras.

5. **Transfer gradients to CPU**: Gradients are transferred from GPU buffers back to CPU pinned memory gradient buffers.

6. **Update on CPU**: The `optimizer.step()` update is performed on CPU. 



