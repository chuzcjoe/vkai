#pragma once

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

namespace vkai {

class LinearLayer : public core::vulkan::VulkanCompute {

public:
  LinearLayer(core::vulkan::VulkanContext *context,
              core::vulkan::VulkanBuffer &input,
              core::vulkan::VulkanBuffer &weights,
              core::vulkan::VulkanBuffer &bias,
              core::vulkan::VulkanBuffer &output, int input_size,
              int output_size, int batch_size = 1);

  void Init() override;

  void Run(const VkCommandBuffer command_buffer);

  bool IsValid() const { return valid_; }

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

  bool valid_ = false;
};

} // namespace vkai
