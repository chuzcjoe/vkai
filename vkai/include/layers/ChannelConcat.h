#pragma once

#include <vector>

#include "layers/Layer.h"

namespace vkai {

// Concatenates 2 to 5 contiguous float32 NCHW tensors along the channel dimension.
class ChannelConcat : public Layer {
 public:
  ChannelConcat(core::vulkan::VulkanContext* context, const std::vector<int>& input_channels,
                int input_height, int input_width, int batch_size = 1);

  void Init() override;

  // The buffer order must match input_channels supplied to the constructor.
  void Execute(const VkCommandBuffer& command_buffer,
               const core::vulkan::VulkanBuffer& input_buffer,
               core::vulkan::VulkanBuffer& output_buffer) override;
  void Execute(const VkCommandBuffer& command_buffer,
               const std::vector<const core::vulkan::VulkanBuffer*>& input_buffers,
               core::vulkan::VulkanBuffer& output_buffer);

  int OutputChannels() const { return uniform_data_.output_channels; }

 protected:
  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;
  const std::string GetPipelineCache() const override;

 private:
  static constexpr int kMaxInputs = 5;

  struct alignas(16) UniformData {
    int input_width;
    int input_height;
    int batch_size;
    int output_channels;
    int input_count;
    int input_channels_0;
    int input_channels_1;
    int input_channels_2;
    int input_channels_3;
    int input_channels_4;
    int reserved_0;
    int reserved_1;
  } uniform_data_;

  static_assert(sizeof(UniformData) == 48);

  core::vulkan::VulkanBuffer uniform_buffer_;
  bool valid_ = false;
};

}  // namespace vkai
