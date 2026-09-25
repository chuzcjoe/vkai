#include "MappedFile.h"

#include <limits>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vkai {

MappedFile::~MappedFile() {
#if defined(_WIN32)
  if (data_ != nullptr) {
    UnmapViewOfFile(data_);
  }
#else
  if (data_ != nullptr) {
    munmap(const_cast<uint8_t*>(data_), size_);
  }
#endif
}

bool MappedFile::Open(const std::string& filename, std::string& error_message) {
#if defined(_WIN32)
  const HANDLE file = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    error_message = "Failed to open file";
    return false;
  }

  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart <= 0 ||
      static_cast<uintmax_t>(file_size.QuadPart) > std::numeric_limits<size_t>::max()) {
    CloseHandle(file);
    error_message = "Invalid file size";
    return false;
  }

  const HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  CloseHandle(file);
  if (mapping == nullptr) {
    error_message = "Failed to create file mapping";
    return false;
  }

  const void* data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(mapping);
  if (data == nullptr) {
    error_message = "Failed to map file";
    return false;
  }

  data_ = static_cast<const uint8_t*>(data);
  size_ = static_cast<size_t>(file_size.QuadPart);
#else
  const int file = open(filename.c_str(), O_RDONLY);
  if (file == -1) {
    error_message = "Failed to open file";
    return false;
  }

  struct stat file_status{};
  if (fstat(file, &file_status) == -1 || file_status.st_size <= 0 ||
      static_cast<uintmax_t>(file_status.st_size) > std::numeric_limits<size_t>::max()) {
    close(file);
    error_message = "Invalid file size";
    return false;
  }

  const size_t file_size = static_cast<size_t>(file_status.st_size);
  void* data = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, file, 0);
  close(file);
  if (data == MAP_FAILED) {
    error_message = "Failed to map file";
    return false;
  }

  data_ = static_cast<const uint8_t*>(data);
  size_ = file_size;
#endif
  return true;
}

}  // namespace vkai
