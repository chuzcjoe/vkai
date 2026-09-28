#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace vkai {

inline constexpr VkMemoryPropertyFlags kHostVisibleMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

template <typename T>
VkDeviceSize GetBufferSize(const std::vector<T>& values) {
  return static_cast<VkDeviceSize>(std::max<std::size_t>(values.size(), 1)) * sizeof(T);
}

}  // namespace vkai
