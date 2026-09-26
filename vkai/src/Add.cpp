#include "Add.h"

#include <cstring>
#include <filesystem>
#include <iostream>

namespace vkai {

Add::Add(core::vulkan::VulkanContext* context, int channel_count, int elements_per_channel,
         int batch_size)
    : Layer(context),
      uniform_data_{
          .channel_count = channel_count,
          .elements_per_channel = elements_per_channel,
          .batch_size = batch_size,
          .element_count = 0,
      },
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
  if (channel_count <= 0 || elements_per_channel <= 0 || batch_size <= 0) {
    std::cerr << "Add dimensions must be positive\n";
    return;
  }
  uniform_data_.element_count = channel_count * elements_per_channel * batch_size;
  uniform_buffer_.MapData(
      [this](void* data) { std::memcpy(data, &uniform_data_, sizeof(UniformData)); });
  valid_ = true;
}

void Add::Init() {
  if (!valid_) {
    std::cerr << "Cannot initialize an invalid Add\n";
    return;
  }
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  vkUpdateDescriptorSets(context_->logical_device, 1, &writes_[0], 0, nullptr);

  const std::string cache = GetPipelineCache();
  if (!cache.empty() && !std::filesystem::exists(cache)) {
    SavePipelineCache(cache);
  }
}

void Add::Execute(const VkCommandBuffer& command_buffer,
                  const core::vulkan::VulkanBuffer& input_buffer,
                  core::vulkan::VulkanBuffer& output_buffer) {
  (void)command_buffer;
  (void)input_buffer;
  (void)output_buffer;
  std::cerr << "Add requires an addend buffer\n";
}

void Add::Execute(const VkCommandBuffer& command_buffer,
                  const core::vulkan::VulkanBuffer& input_buffer,
                  const core::vulkan::VulkanBuffer& addend_buffer,
                  core::vulkan::VulkanBuffer& output_buffer) {
  if (!valid_ || pipeline == VK_NULL_HANDLE) {
    std::cerr << "Cannot run an invalid or uninitialized Add\n";
    return;
  }
  const VkDeviceSize input_size =
      static_cast<VkDeviceSize>(uniform_data_.element_count) * sizeof(float);
  const VkDeviceSize addend_size =
      static_cast<VkDeviceSize>(uniform_data_.channel_count) * sizeof(float);
  if (input_buffer.Size() < input_size || addend_buffer.Size() < addend_size ||
      output_buffer.Size() < input_size) {
    std::cerr << "Add input, addend, or output buffer is too small\n";
    return;
  }
  const VkDescriptorBufferInfo buffer_infos[] = {
      {.buffer = input_buffer.buffer, .offset = 0, .range = input_buffer.Size()},
      {.buffer = addend_buffer.buffer, .offset = 0, .range = addend_buffer.Size()},
      {.buffer = output_buffer.buffer, .offset = 0, .range = output_buffer.Size()},
  };
  const VkWriteDescriptorSet descriptor_writes[] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet = descriptor_set_,
       .dstBinding = 1,
       .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &buffer_infos[0]},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet = descriptor_set_,
       .dstBinding = 2,
       .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &buffer_infos[1]},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
       .dstSet = descriptor_set_,
       .dstBinding = 3,
       .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &buffer_infos[2]},
  };
  vkUpdateDescriptorSets(context_->logical_device, 3, descriptor_writes, 0, nullptr);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  constexpr uint32_t kLocalSize = 256;
  vkCmdDispatch(command_buffer,
                (static_cast<uint32_t>(uniform_data_.element_count) + kLocalSize - 1) / kLocalSize,
                1, 1);
}

std::vector<core::vulkan::BindingInfo> Add::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& Add::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "Add.comp.spv"
      ;
  return shader_code;
}

const std::string Add::GetPipelineCache() const {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  if (cache_dir.empty()) {
    return "";
  }
  return cache_dir + "/add.cache";
}

}  // namespace vkai
