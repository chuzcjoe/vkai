#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace vkai {

// Owns a read-only memory mapping of a file.
class MappedFile {
 public:
  MappedFile() = default;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  // Maps filename for read-only access. On failure, error_message describes the cause.
  bool Open(const std::string& filename, std::string& error_message);

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
};

}  // namespace vkai
