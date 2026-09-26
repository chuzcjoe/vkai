#pragma once

#include "Layer.h"

namespace vkai {

// Adds a per-channel tensor to an NCHW input. The addend has channel_count elements and is
// broadcast across every spatial element and batch item.
class Add : public Layer {
 public:
  Add(core::vulkan::VulkanContext* context, int channel_count, int elements_per_channel,
      int batch_size = 1);

  void Init() override;

  // Add requires a second input buffer; use the overload below for execution.
  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;

  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               const core::vulkan::VulkanBuffer& addend_buffer,
               core::vulkan::VulkanBuffer& output_buffer);

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;

  const std::vector<uint32_t>& LoadShaderCode() const override;

  const std::string GetPipelineCache() const override;

 private:
  struct UniformData {
    int channel_count;
    int elements_per_channel;
    int batch_size;
    int element_count;
  } uniform_data_;

  core::vulkan::VulkanBuffer uniform_buffer_;
  bool valid_ = false;
};

}  // namespace vkai
