#pragma once

#include "Layer.h"

namespace vkai {

class Linear : public Layer {

public:
  Linear(core::vulkan::VulkanContext *context,
         core::vulkan::VulkanBuffer &input, core::vulkan::VulkanBuffer &weights,
         core::vulkan::VulkanBuffer &bias, core::vulkan::VulkanBuffer &output,
         int input_size, int output_size, int batch_size = 1);

  void Init() override;

  void Execute(const VkCommandBuffer &command_buffer) override;

protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t> &LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

private:
  core::vulkan::VulkanBuffer &input_buffer_;
  core::vulkan::VulkanBuffer &weights_buffer_;
  core::vulkan::VulkanBuffer &bias_buffer_;
  core::vulkan::VulkanBuffer &output_buffer_;

  core::vulkan::VulkanBuffer uniform_buffer_;
  struct UniformData {
    int input_size;
    int output_size;
    int batch_size;
  } uniform_data_;
};

} // namespace vkai
