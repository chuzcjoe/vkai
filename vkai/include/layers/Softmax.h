#pragma once

#include "layers/Layer.h"

namespace vkai {

enum class SoftmaxMode {
  // Treat each batch item as one contiguous vector. This is the existing behavior.
  kContiguousPerBatch,
  // Treat each NCHW spatial position as one vector across the channel dimension.
  kChannelNCHW,
};

class Softmax : public Layer {
 public:
  // Computes softmax over each contiguous batch vector.
  Softmax(core::vulkan::VulkanContext* context, int input_size, int batch_size = 1);
  // Computes softmax over C for every NCHW location. mode must be kChannelNCHW.
  Softmax(core::vulkan::VulkanContext* context, int channel_count, int input_height,
          int input_width, int batch_size, SoftmaxMode mode);

  void Init() override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

 private:
  core::vulkan::VulkanBuffer uniform_buffer_;

  struct alignas(16) UniformData {
    int channel_count;
    int batch_size;
    int input_height;
    int input_width;
    int mode;
    int reserved_0;
    int reserved_1;
    int reserved_2;
  } uniform_data_;

  bool valid_ = false;
};

}  // namespace vkai
