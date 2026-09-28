#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <vector>

namespace vkai {

inline constexpr VkMemoryPropertyFlags kHostVisibleMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

inline VkDeviceSize BufferSize(const std::vector<float>& values) {
  return std::max<size_t>(values.size(), 1) * sizeof(float);
}

}  // namespace vkai
