#pragma once

#include "Layer.h"

namespace vkai {

// Max pooling over NCHW tensors without padding.
class MaxPool2D : public Layer {
 public:
  MaxPool2D(core::vulkan::VulkanContext* context, int input_channels, int input_height,
            int input_width, int kernel_height, int kernel_width, int stride_height,
            int stride_width, int batch_size = 1);

  void Init() override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

  int OutputHeight() const { return uniform_data_.output_height; }
  int OutputWidth() const { return uniform_data_.output_width; }

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

 private:
  struct alignas(16) UniformData {
    int input_width;
    int input_height;
    int input_channels;
    int batch_size;

    int output_width;
    int output_height;
    int kernel_width;
    int kernel_height;

    int stride_width;
    int stride_height;
    int reserved_0;
    int reserved_1;
  };

  static_assert(sizeof(UniformData) == 48);

  core::vulkan::VulkanBuffer uniform_buffer_;
  UniformData uniform_data_;
  bool valid_ = false;
};

}  // namespace vkai
