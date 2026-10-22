#include "engine/framework/assets/gguf_metadata.h"

#include <gguf.h>

#include <stdexcept>

namespace engine::assets {

GgufMetadata::GgufMetadata(const std::filesystem::path & path) : path_(path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    ctx_.reset(gguf_init_from_file(path.string().c_str(), params));
    if (!ctx_) throw std::runtime_error("failed to read GGUF metadata: " + path.string());
}

void GgufMetadata::Deleter::operator()(gguf_context * ctx) const noexcept { gguf_free(ctx); }

bool GgufMetadata::has(const char * key) const { return gguf_find_key(ctx_.get(), key) >= 0; }

int64_t GgufMetadata::require_int(const char * key) const {
    const int64_t id = require_key(key);
    switch (gguf_get_kv_type(ctx_.get(), id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx_.get(), id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx_.get(), id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx_.get(), id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx_.get(), id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx_.get(), id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx_.get(), id);
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ctx_.get(), id));
        case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx_.get(), id);
        default: throw std::runtime_error(std::string("GGUF key is not an integer: ") + key);
    }
}

float GgufMetadata::require_float(const char * key) const {
    const int64_t id = require_key(key);
    switch (gguf_get_kv_type(ctx_.get(), id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx_.get(), id);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ctx_.get(), id));
        default: throw std::runtime_error(std::string("GGUF key is not a float: ") + key);
    }
}

std::string GgufMetadata::require_string(const char * key) const {
    const int64_t id = require_key(key);
    if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("GGUF key is not a string: ") + key);
    }
    return gguf_get_val_str(ctx_.get(), id);
}

std::vector<int32_t> GgufMetadata::require_int_array(const char * key) const {
    const int64_t id = require_key(key);
    if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(std::string("GGUF key is not an array: ") + key);
    }
    const size_t n = gguf_get_arr_n(ctx_.get(), id);
    const void * data = gguf_get_arr_data(ctx_.get(), id);
    std::vector<int32_t> out(n);
    switch (gguf_get_arr_type(ctx_.get(), id)) {
        case GGUF_TYPE_INT32:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<const int32_t *>(data)[i];
            break;
        case GGUF_TYPE_UINT32:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(static_cast<const uint32_t *>(data)[i]);
            break;
        case GGUF_TYPE_INT64:
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<int32_t>(static_cast<const int64_t *>(data)[i]);
            break;
        default:
            throw std::runtime_error(std::string("GGUF array is not integer: ") + key);
    }
    return out;
}

std::vector<std::string> GgufMetadata::require_string_array(const char * key) const {
    const int64_t id = require_key(key);
    if (gguf_get_kv_type(ctx_.get(), id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx_.get(), id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("GGUF key is not a string array: ") + key);
    }
    const size_t n = gguf_get_arr_n(ctx_.get(), id);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(ctx_.get(), id, i));
    return out;
}

int64_t GgufMetadata::require_key(const char * key) const {
    const int64_t id = gguf_find_key(ctx_.get(), key);
    if (id < 0) throw std::runtime_error(std::string("GGUF is missing key ") + key + ": " + path_.string());
    return id;
}

}  // namespace engine::assets
