#include "engine/framework/assets/onnx_model.h"

#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::assets::onnx {
namespace {

// Protocol buffers wire format; field numbers from onnx.proto (ModelProto, GraphProto, NodeProto,
// TensorProto, StringStringEntryProto).
constexpr uint32_t kVarint = 0;
constexpr uint32_t kFixed64 = 1;
constexpr uint32_t kLengthDelimited = 2;
constexpr uint32_t kFixed32 = 5;

class Reader {
public:
    explicit Reader(const std::filesystem::path & path) : path_(path), in_(path, std::ios::binary) {
        if (!in_) throw std::runtime_error("cannot open ONNX file: " + path.string());
        in_.seekg(0, std::ios::end);
        size_ = static_cast<uint64_t>(in_.tellg());
        in_.seekg(0, std::ios::beg);
    }

    uint64_t size() const noexcept { return size_; }
    uint64_t position() const noexcept { return position_; }

    uint64_t varint() {
        uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            const uint8_t byte = next();
            value |= static_cast<uint64_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) return value;
        }
        fail("varint is too long");
    }

    std::string bytes(uint64_t count) {
        require(count);
        std::string out(static_cast<size_t>(count), '\0');
        in_.read(out.data(), static_cast<std::streamsize>(count));
        if (!in_) fail("read past the end of the file");
        position_ += count;
        return out;
    }

    void skip(uint64_t count) {
        require(count);
        position_ += count;
        in_.seekg(static_cast<std::streamoff>(position_), std::ios::beg);
    }

    void skip_field(uint32_t wire_type) {
        switch (wire_type) {
            case kVarint: (void) varint(); return;
            case kFixed64: skip(8); return;
            case kLengthDelimited: skip(varint()); return;
            case kFixed32: skip(4); return;
            default: fail("unsupported protobuf wire type " + std::to_string(wire_type));
        }
    }

    [[noreturn]] void fail(const std::string & what) const {
        throw std::runtime_error("malformed ONNX file " + path_.string() + ": " + what);
    }

private:
    uint8_t next() {
        require(1);
        const int value = in_.get();
        if (value == std::char_traits<char>::eof()) fail("read past the end of the file");
        ++position_;
        return static_cast<uint8_t>(value);
    }

    void require(uint64_t count) const {
        if (count > size_ - position_) fail("a field runs past the end of the file");
    }

    std::filesystem::path path_;
    std::ifstream in_;
    uint64_t size_ = 0;
    uint64_t position_ = 0;
};

// Calls `field(number, wire_type)` for every field of the message that ends at `end`.
template <typename Field>
void read_message(Reader & reader, uint64_t end, Field && field) {
    while (reader.position() < end) {
        const uint64_t key = reader.varint();
        field(static_cast<uint32_t>(key >> 3), static_cast<uint32_t>(key & 7));
    }
    if (reader.position() != end) reader.fail("a message overruns its length");
}

uint64_t message_end(Reader & reader) {
    const uint64_t length = reader.varint();
    if (length > reader.size() - reader.position()) reader.fail("a message runs past the end of the file");
    return reader.position() + length;
}

const char * dtype_name(uint64_t data_type) {
    switch (data_type) {
        case 1: return "F32";
        case 2: return "U8";
        case 3: return "I8";
        case 7: return "I64";
        case 10: return "F16";
        case 16: return "BF16";
        default: return nullptr;
    }
}

size_t dtype_bytes(const std::string & dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "I64") return 8;
    if (dtype == "I8" || dtype == "U8") return 1;
    return 2;
}

int64_t element_count(const std::vector<int64_t> & dims) {
    int64_t total = 1;
    for (const int64_t dim : dims) total *= dim;
    return total;
}

Initializer read_initializer(Reader & reader, uint64_t end) {
    Initializer out;
    uint64_t data_type = 0;
    bool has_data = false;
    bool external = false;
    read_message(reader, end, [&](uint32_t field, uint32_t wire) {
        if (field == 1 && wire == kVarint) {
            out.dims.push_back(static_cast<int64_t>(reader.varint()));
        } else if (field == 1 && wire == kLengthDelimited) {
            const uint64_t packed_end = message_end(reader);
            while (reader.position() < packed_end) out.dims.push_back(static_cast<int64_t>(reader.varint()));
        } else if (field == 2 && wire == kVarint) {
            data_type = reader.varint();
        } else if (field == 8 && wire == kLengthDelimited) {
            out.name = reader.bytes(reader.varint());
        } else if ((field == 9 || field == 4) && wire == kLengthDelimited) {
            // raw_data, or float_data packed as little-endian floats: the same bytes either way.
            out.bytes = reader.varint();
            out.offset = reader.position();
            has_data = true;
            reader.skip(out.bytes);
        } else if (field == 14 && wire == kVarint) {
            external = reader.varint() == 1;
        } else {
            reader.skip_field(wire);
        }
    });
    const char * dtype = dtype_name(data_type);
    if (external) reader.fail("initializer " + out.name + " keeps its data in an external file");
    if (dtype == nullptr) reader.fail("initializer " + out.name + " has unsupported data type " + std::to_string(data_type));
    out.dtype = dtype;
    // a scalar: one value, which the other weight files keep as shape [1]
    if (out.dims.empty()) out.dims = {1};
    if (!has_data) reader.fail("initializer " + out.name + " stores no packed data");
    if (out.bytes != static_cast<uint64_t>(element_count(out.dims)) * dtype_bytes(out.dtype)) {
        reader.fail("initializer " + out.name + " data size does not match its shape");
    }
    return out;
}

// AttributeProto: name (1) and, for an integer attribute, i (3).
void read_attribute(Reader & reader, uint64_t end, Node & node) {
    std::string name;
    std::optional<int64_t> value;
    read_message(reader, end, [&](uint32_t field, uint32_t wire) {
        if (field == 1 && wire == kLengthDelimited) {
            name = reader.bytes(reader.varint());
        } else if (field == 3 && wire == kVarint) {
            value = static_cast<int64_t>(reader.varint());
        } else {
            reader.skip_field(wire);
        }
    });
    if (value.has_value() && !name.empty()) node.int_attributes[name] = *value;
}

Node read_node(Reader & reader, uint64_t end) {
    Node out;
    read_message(reader, end, [&](uint32_t field, uint32_t wire) {
        if (wire != kLengthDelimited) {
            reader.skip_field(wire);
        } else if (field == 1) {
            out.inputs.push_back(reader.bytes(reader.varint()));
        } else if (field == 2) {
            out.outputs.push_back(reader.bytes(reader.varint()));
        } else if (field == 3) {
            out.name = reader.bytes(reader.varint());
        } else if (field == 4) {
            out.op_type = reader.bytes(reader.varint());
        } else if (field == 5) {
            read_attribute(reader, message_end(reader), out);
        } else {
            reader.skip_field(wire);
        }
    });
    return out;
}

void read_graph(Reader & reader, uint64_t end, Model & model) {
    read_message(reader, end, [&](uint32_t field, uint32_t wire) {
        if (field == 1 && wire == kLengthDelimited) {
            model.nodes.push_back(read_node(reader, message_end(reader)));
        } else if (field == 5 && wire == kLengthDelimited) {
            auto initializer = read_initializer(reader, message_end(reader));
            std::string name = initializer.name;
            model.initializers.emplace(std::move(name), std::move(initializer));
        } else {
            reader.skip_field(wire);
        }
    });
}

std::pair<std::string, std::string> read_entry(Reader & reader, uint64_t end) {
    std::pair<std::string, std::string> out;
    read_message(reader, end, [&](uint32_t field, uint32_t wire) {
        if (field == 1 && wire == kLengthDelimited) {
            out.first = reader.bytes(reader.varint());
        } else if (field == 2 && wire == kLengthDelimited) {
            out.second = reader.bytes(reader.varint());
        } else {
            reader.skip_field(wire);
        }
    });
    return out;
}

class RoutedSource final : public TensorSource {
public:
    RoutedSource(
        std::shared_ptr<const Model> model,
        std::unordered_map<std::string, Route> routes,
        std::unordered_map<std::string, Computed> computed)
        : model_(std::move(model)), routes_(std::move(routes)), computed_(std::move(computed)) {
        for (const auto & [name, route] : routes_) {
            const auto & initializer = require_initializer(route.initializer);
            if (route.transpose && initializer.dims.size() != 2) {
                throw std::runtime_error("ONNX initializer " + route.initializer + " to transpose is not a matrix");
            }
            if (!route.scale.empty()) check_quantized(route, initializer);
        }
        for (const auto & [name, tensor] : computed_) {
            if (static_cast<int64_t>(tensor.values.size()) != element_count(tensor.dims)) {
                throw std::runtime_error("computed tensor " + name + " does not match its shape");
            }
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return model_->path; }

    bool has_tensor(std::string_view name) const noexcept override {
        const std::string key(name);
        return routes_.count(key) != 0 || computed_.count(key) != 0;
    }

    TensorMetadata require_metadata(std::string_view name) const override {
        const std::string key(name);
        if (const auto it = computed_.find(key); it != computed_.end()) {
            return {key, "F32", it->second.dims};
        }
        const auto & route = require_route(key);
        const auto & initializer = require_initializer(route.initializer);
        return {key, route.scale.empty() ? initializer.dtype : "F32", routed_dims(route, initializer)};
    }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        out.reserve(routes_.size() + computed_.size());
        for (const auto & [name, route] : routes_) out.push_back(require_metadata(name));
        for (const auto & [name, tensor] : computed_) out.push_back({name, "F32", tensor.dims});
        return out;
    }

    RawTensorData require_tensor_data(std::string_view name) const override {
        const std::string key(name);
        RawTensorData out;
        out.metadata = require_metadata(key);
        if (const auto it = computed_.find(key); it != computed_.end()) {
            out.bytes.resize(it->second.values.size() * sizeof(float));
            std::memcpy(out.bytes.data(), it->second.values.data(), out.bytes.size());
            return out;
        }
        const auto & route = require_route(key);
        const auto & initializer = require_initializer(route.initializer);
        out.bytes = read_bytes(initializer);
        if (!route.scale.empty()) out.bytes = dequantized(out.bytes, route, initializer);
        if (route.transpose) out.bytes = transposed(out.bytes, initializer, out.metadata.dtype);
        return out;
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto raw = require_tensor_data(name);
        if (expected_shape.has_value() && *expected_shape != raw.metadata.shape) {
            throw std::runtime_error("ONNX tensor shape mismatch for " + std::string(name));
        }
        const size_t count = static_cast<size_t>(element_count(raw.metadata.shape));
        std::vector<float> values(count);
        if (raw.metadata.dtype == "F32") {
            std::memcpy(values.data(), raw.bytes.data(), count * sizeof(float));
        } else if (raw.metadata.dtype == "F16") {
            ggml_fp16_to_fp32_row(
                reinterpret_cast<const ggml_fp16_t *>(raw.bytes.data()), values.data(), static_cast<int64_t>(count));
        } else if (raw.metadata.dtype == "BF16") {
            ggml_bf16_to_fp32_row(
                reinterpret_cast<const ggml_bf16_t *>(raw.bytes.data()), values.data(), static_cast<int64_t>(count));
        } else {
            throw std::runtime_error("ONNX tensor is not floating point: " + std::string(name));
        }
        return values;
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        const auto raw = require_tensor_data(name);
        if (raw.metadata.dtype != "I64" || element_count(raw.metadata.shape) != 1) {
            throw std::runtime_error("ONNX tensor is not an i64 scalar: " + std::string(name));
        }
        int64_t value = 0;
        std::memcpy(&value, raw.bytes.data(), sizeof(value));
        return value;
    }

private:
    const Route & require_route(const std::string & name) const {
        const auto it = routes_.find(name);
        if (it == routes_.end()) {
            throw std::runtime_error("missing tensor in " + model_->path.filename().string() + ": " + name);
        }
        return it->second;
    }

    const Initializer & require_initializer(const std::string & name) const {
        const auto it = model_->initializers.find(name);
        if (it == model_->initializers.end()) {
            throw std::runtime_error("missing ONNX initializer in " + model_->path.filename().string() + ": " + name);
        }
        return it->second;
    }

    static std::vector<int64_t> routed_dims(const Route & route, const Initializer & initializer) {
        if (!route.transpose) return initializer.dims;
        return {initializer.dims[1], initializer.dims[0]};
    }

    std::vector<std::byte> read_bytes(const Initializer & initializer) const {
        std::ifstream in(model_->path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open ONNX file: " + model_->path.string());
        in.seekg(static_cast<std::streamoff>(initializer.offset), std::ios::beg);
        std::vector<std::byte> out(static_cast<size_t>(initializer.bytes));
        in.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size()));
        if (!in) throw std::runtime_error("cannot read ONNX initializer " + initializer.name);
        return out;
    }

    void check_quantized(const Route & route, const Initializer & initializer) const {
        if (initializer.dtype != "I8" && initializer.dtype != "U8") {
            throw std::runtime_error("ONNX initializer " + route.initializer + " to dequantize is not 8-bit");
        }
        const auto & scale = require_initializer(route.scale);
        const int64_t count = element_count(scale.dims);
        const int64_t rank = static_cast<int64_t>(initializer.dims.size());
        const int64_t axis = route.axis < 0 ? route.axis + rank : route.axis;
        if (scale.dtype != "F32" || (count != 1 && (axis < 0 || axis >= rank || count != initializer.dims[axis]))) {
            throw std::runtime_error("ONNX scale " + route.scale + " does not fit " + route.initializer);
        }
        if (!route.zero_point.empty()) {
            const auto & zero_point = require_initializer(route.zero_point);
            if (zero_point.dtype != initializer.dtype || element_count(zero_point.dims) != count) {
                throw std::runtime_error("ONNX zero point " + route.zero_point + " does not fit " + route.initializer);
            }
        }
    }

    std::vector<std::byte> dequantized(
        const std::vector<std::byte> & bytes, const Route & route, const Initializer & initializer) const {
        const auto & scale_init = require_initializer(route.scale);
        const auto scale_bytes = read_bytes(scale_init);
        std::vector<float> scale(static_cast<size_t>(element_count(scale_init.dims)));
        std::memcpy(scale.data(), scale_bytes.data(), scale.size() * sizeof(float));
        std::vector<std::byte> zero(scale.size(), std::byte{0});
        if (!route.zero_point.empty()) zero = read_bytes(require_initializer(route.zero_point));
        const bool is_signed = initializer.dtype == "I8";
        const auto as_int = [is_signed](std::byte b) {
            return is_signed ? static_cast<int32_t>(static_cast<int8_t>(b)) : static_cast<int32_t>(static_cast<uint8_t>(b));
        };
        const int64_t rank = static_cast<int64_t>(initializer.dims.size());
        const int64_t axis = route.axis < 0 ? route.axis + rank : route.axis;
        int64_t inner = 1;
        for (int64_t d = axis + 1; d < rank; ++d) inner *= initializer.dims[static_cast<size_t>(d)];
        const int64_t channels = scale.size() == 1 ? 1 : initializer.dims[static_cast<size_t>(axis)];
        std::vector<float> values(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) {
            const size_t c = channels == 1 ? 0 : static_cast<size_t>((static_cast<int64_t>(i) / inner) % channels);
            values[i] = static_cast<float>(as_int(bytes[i]) - as_int(zero[c])) * scale[c];
        }
        std::vector<std::byte> out(values.size() * sizeof(float));
        std::memcpy(out.data(), values.data(), out.size());
        return out;
    }

    static std::vector<std::byte> transposed(
        const std::vector<std::byte> & bytes, const Initializer & initializer, const std::string & dtype) {
        const size_t rows = static_cast<size_t>(initializer.dims[0]);
        const size_t cols = static_cast<size_t>(initializer.dims[1]);
        const size_t width = dtype_bytes(dtype);
        std::vector<std::byte> out(bytes.size());
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                std::memcpy(out.data() + (c * rows + r) * width, bytes.data() + (r * cols + c) * width, width);
            }
        }
        return out;
    }

    std::shared_ptr<const Model> model_;
    std::unordered_map<std::string, Route> routes_;
    std::unordered_map<std::string, Computed> computed_;
};

class JoinedSource final : public TensorSource {
public:
    explicit JoinedSource(std::vector<std::shared_ptr<const TensorSource>> sources) : sources_(std::move(sources)) {
        for (size_t i = 0; i < sources_.size(); ++i) {
            for (const auto & tensor : sources_[i]->tensors()) {
                if (!owners_.emplace(tensor.name, i).second) {
                    throw std::runtime_error("ONNX tensor " + tensor.name + " comes from two files");
                }
            }
        }
    }

    const std::filesystem::path & source_path() const noexcept override { return sources_.front()->source_path(); }
    bool has_tensor(std::string_view name) const noexcept override { return owners_.count(std::string(name)) != 0; }
    TensorMetadata require_metadata(std::string_view name) const override { return owner(name).require_metadata(name); }

    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        for (const auto & source : sources_) {
            auto part = source->tensors();
            out.insert(out.end(), part.begin(), part.end());
        }
        return out;
    }

    RawTensorData require_tensor_data(std::string_view name) const override { return owner(name).require_tensor_data(name); }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return owner(name).require_f32(name, expected_shape);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override { return owner(name).require_i64_scalar(name); }

private:
    const TensorSource & owner(std::string_view name) const {
        const auto it = owners_.find(std::string(name));
        if (it == owners_.end()) throw std::runtime_error("missing ONNX tensor: " + std::string(name));
        return *sources_[it->second];
    }

    std::vector<std::shared_ptr<const TensorSource>> sources_;
    std::unordered_map<std::string, size_t> owners_;
};

}  // namespace

bool is_file(const std::filesystem::path & path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) return false;
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return extension == ".onnx";
}

Model read_model(const std::filesystem::path & path) {
    Reader reader(path);
    Model model;
    model.path = path;
    bool has_graph = false;
    read_message(reader, reader.size(), [&](uint32_t field, uint32_t wire) {
        if (field == 7 && wire == kLengthDelimited) {
            read_graph(reader, message_end(reader), model);
            has_graph = true;
        } else if (field == 14 && wire == kLengthDelimited) {
            model.metadata.insert(read_entry(reader, message_end(reader)));
        } else {
            reader.skip_field(wire);
        }
    });
    if (!has_graph) reader.fail("no graph");
    return model;
}

std::shared_ptr<const TensorSource> open_routed(
    std::shared_ptr<const Model> model,
    std::unordered_map<std::string, Route> routes,
    std::unordered_map<std::string, Computed> computed) {
    if (model == nullptr) throw std::runtime_error("routed ONNX tensor source requires a model");
    return std::make_shared<RoutedSource>(std::move(model), std::move(routes), std::move(computed));
}

std::shared_ptr<const TensorSource> join(std::vector<std::shared_ptr<const TensorSource>> sources) {
    if (sources.empty()) throw std::runtime_error("joined ONNX tensor source requires a source");
    return std::make_shared<JoinedSource>(std::move(sources));
}

}  // namespace engine::assets::onnx
