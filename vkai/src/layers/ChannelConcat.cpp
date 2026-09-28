#include "layers/ChannelConcat.h"

#include <cstring>
#include <filesystem>
#include <iostream>

namespace vkai {

ChannelConcat::ChannelConcat(core::vulkan::VulkanContext* context,
                             const std::vector<int>& input_channels, int input_height,
                             int input_width, int batch_size)
    : Layer(context),
      uniform_data_{.input_width = input_width,
                    .input_height = input_height,
                    .batch_size = batch_size,
                    .output_channels = 0,
                    .input_count = static_cast<int>(input_channels.size()),
                    .input_channels_0 = 0,
                    .input_channels_1 = 0,
                    .input_channels_2 = 0,
                    .input_channels_3 = 0,
                    .input_channels_4 = 0,
                    .reserved_0 = 0,
                    .reserved_1 = 0},
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
  if (input_channels.size() < 2 || input_channels.size() > kMaxInputs || input_height <= 0 ||
      input_width <= 0 || batch_size <= 0) {
    std::cerr << "ChannelConcat requires 2 to 5 inputs and positive dimensions\n";
    return;
  }
  int* const channels[] = {&uniform_data_.input_channels_0, &uniform_data_.input_channels_1,
                           &uniform_data_.input_channels_2, &uniform_data_.input_channels_3,
                           &uniform_data_.input_channels_4};
  for (size_t index = 0; index < input_channels.size(); ++index) {
    if (input_channels[index] <= 0) {
      std::cerr << "ChannelConcat channel counts must be positive\n";
      return;
    }
    *channels[index] = input_channels[index];
    uniform_data_.output_channels += input_channels[index];
  }
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void ChannelConcat::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid ChannelConcat\n";
    return;
  }
  VulkanCompute::Init();
  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);
  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) SavePipelineCache(cache);
}

void ChannelConcat::Execute(const VkCommandBuffer& command_buffer,
                            const core::vulkan::VulkanBuffer& input_buffer,
                            core::vulkan::VulkanBuffer& output_buffer) {
  (void)command_buffer;
  (void)input_buffer;
  (void)output_buffer;
  std::cerr << "ChannelConcat requires all input buffers\n";
}

void ChannelConcat::Execute(const VkCommandBuffer& command_buffer,
                            const std::vector<const core::vulkan::VulkanBuffer*>& input_buffers,
                            core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized ChannelConcat\n";
    return;
  }
  if (input_buffers.size() != static_cast<size_t>(uniform_data_.input_count)) {
    std::cerr << "ChannelConcat input buffer count does not match channel counts\n";
    return;
  }
  const int channels[] = {uniform_data_.input_channels_0, uniform_data_.input_channels_1,
                          uniform_data_.input_channels_2, uniform_data_.input_channels_3,
                          uniform_data_.input_channels_4};
  const VkDeviceSize elements_per_channel = static_cast<VkDeviceSize>(uniform_data_.batch_size) *
                                            uniform_data_.input_height * uniform_data_.input_width;
  VkDescriptorBufferInfo buffer_infos[kMaxInputs + 1] = {};
  VkWriteDescriptorSet descriptor_writes[kMaxInputs + 1] = {};
  for (int index = 0; index < kMaxInputs; ++index) {
    const int source_index = index < uniform_data_.input_count ? index : 0;
    const auto* buffer = input_buffers[source_index];
    const VkDeviceSize required_size =
        elements_per_channel * channels[source_index] * sizeof(float);
    if (buffer == nullptr || buffer->Size() < required_size) {
      std::cerr << "ChannelConcat input buffer is too small\n";
      return;
    }
    buffer_infos[index] = {.buffer = buffer->buffer, .offset = 0, .range = buffer->Size()};
    descriptor_writes[index] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = descriptor_set_,
                                .dstBinding = static_cast<uint32_t>(index + 1),
                                .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .pBufferInfo = &buffer_infos[index]};
  }
  const VkDeviceSize output_size =
      elements_per_channel * uniform_data_.output_channels * sizeof(float);
  if (output_buffer.Size() < output_size) {
    std::cerr << "ChannelConcat output buffer is too small\n";
    return;
  }
  buffer_infos[kMaxInputs] = {
      .buffer = output_buffer.buffer, .offset = 0, .range = output_buffer.Size()};
  descriptor_writes[kMaxInputs] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                   .dstSet = descriptor_set_,
                                   .dstBinding = kMaxInputs + 1,
                                   .descriptorCount = 1,
                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                   .pBufferInfo = &buffer_infos[kMaxInputs]};
  vkUpdateDescriptorSets(context_->logical_device, kMaxInputs + 1, descriptor_writes, 0, nullptr);
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  const uint32_t output_elements = static_cast<uint32_t>(output_size / sizeof(float));
  constexpr uint32_t kLocalSize = 256;
  vkCmdDispatch(command_buffer, (output_elements + kLocalSize - 1) / kLocalSize, 1, 1);
}

std::vector<core::vulkan::BindingInfo> ChannelConcat::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& ChannelConcat::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "ChannelConcat.comp.spv"
      ;
  return shader_code;
}

const std::string ChannelConcat::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  return cache_dir.empty() ? "" : cache_dir + "/channel_concat.cache";
}

}  // namespace vkai
