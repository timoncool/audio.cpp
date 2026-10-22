#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// An ONNX file read as a weight container: its initializers (raw little-endian data inside the file), the
// names of the nodes that consume them, and the model metadata. A model loads such a file in place of its
// own package by naming each tensor it needs after an initializer, transposed where the exporter turned a
// linear layer into MatMul against the [in, out] weight.
namespace engine::assets::onnx {

struct Initializer {
    std::string name;
    std::string dtype;
    std::vector<int64_t> dims;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct Node {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, int64_t> int_attributes;
};

struct Model {
    std::filesystem::path path;
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, Initializer> initializers;
    std::vector<Node> nodes;
};

bool is_file(const std::filesystem::path & path);

Model read_model(const std::filesystem::path & path);

// A weight read as it is stored, transposed, or dequantized as DequantizeLinear does it:
// (q - zero_point) * scale, per slice of `axis` when scale is a vector.
struct Route {
    std::string initializer;
    bool transpose = false;
    std::string scale;
    std::string zero_point;
    int64_t axis = 1;
};

struct Computed {
    std::vector<int64_t> dims;
    std::vector<float> values;
};

// The routed initializers under their model names, plus tensors the caller computed itself.
std::shared_ptr<const TensorSource> open_routed(
    std::shared_ptr<const Model> model,
    std::unordered_map<std::string, Route> routes,
    std::unordered_map<std::string, Computed> computed);

// Several routed sources as one; a name must belong to exactly one of them.
std::shared_ptr<const TensorSource> join(std::vector<std::shared_ptr<const TensorSource>> sources);

}  // namespace engine::assets::onnx
