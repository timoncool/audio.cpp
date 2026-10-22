#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;

namespace engine::assets {

// The key-value metadata of a GGUF file, read without its tensors.
class GgufMetadata {
public:
    explicit GgufMetadata(const std::filesystem::path & path);

    bool has(const char * key) const;
    int64_t require_int(const char * key) const;
    float require_float(const char * key) const;
    std::string require_string(const char * key) const;
    std::vector<int32_t> require_int_array(const char * key) const;
    std::vector<std::string> require_string_array(const char * key) const;

private:
    int64_t require_key(const char * key) const;

    struct Deleter {
        void operator()(gguf_context * ctx) const noexcept;
    };
    std::filesystem::path path_;
    std::unique_ptr<gguf_context, Deleter> ctx_;
};

}  // namespace engine::assets
