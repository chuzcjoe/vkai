#pragma once

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

namespace vkai {

class Relu : public core::vulkan::VulkanCompute {
public:
  Relu(core::vulkan::VulkanContext *context, core::vulkan::VulkanBuffer &input,
       core::vulkan::VulkanBuffer &output, int element_count);

  void Init() override;

  void Run(VkCommandBuffer command_buffer);

protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t> &LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

private:
  core::vulkan::VulkanBuffer &input_buffer_;
  core::vulkan::VulkanBuffer &output_buffer_;
  core::vulkan::VulkanBuffer uniform_buffer_;

  struct UniformData {
    int element_count;
  } uniform_data_;
};

} // namespace vkai
