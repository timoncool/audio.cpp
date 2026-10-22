#include "engine/models/supertonic/assets.h"

#include "engine/framework/assets/onnx_model.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/io/config.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::supertonic {
namespace json = engine::io::json;
namespace {

SupertonicConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("tts_config");
    SupertonicConfig config;
    const auto & ae = root.require("ae");
    const auto & ttl = root.require("ttl");
    config.sample_rate = static_cast<int>(json::require_i64(ae, "sample_rate"));
    config.base_chunk_size = json::require_i64(ae, "base_chunk_size");
    config.chunk_compress_factor = json::require_i64(ttl, "chunk_compress_factor");
    config.latent_dim = json::require_i64(ttl, "latent_dim");
    engine::io::require_positive(config.sample_rate, "sample_rate");
    engine::io::require_positive(config.base_chunk_size, "base_chunk_size");
    engine::io::require_positive(config.chunk_compress_factor, "chunk_compress_factor");
    engine::io::require_positive(config.latent_dim, "latent_dim");
    return config;
}

std::unordered_map<uint32_t, int64_t> parse_unicode_indexer(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("unicode_indexer");
    std::unordered_map<uint32_t, int64_t> out;
    if (root.is_array()) {
        const auto & values = root.as_array();
        out.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const int64_t token = values[i].as_i64();
            if (token >= 0) {
                out.emplace(static_cast<uint32_t>(i), token);
            }
        }
    } else if (root.is_object()) {
        out.reserve(root.as_object().size());
        for (const auto & [key, value] : root.as_object()) {
            size_t parsed = 0;
            const auto codepoint = static_cast<uint32_t>(std::stoul(key, &parsed, 10));
            if (parsed != key.size()) {
                throw std::runtime_error("Supertonic unicode indexer contains a non-numeric codepoint key: " + key);
            }
            const int64_t token = value.as_i64();
            if (token >= 0) {
                out.emplace(codepoint, token);
            }
        }
    } else {
        throw std::runtime_error("Supertonic unicode indexer must be an array or object");
    }
    if (out.empty()) {
        throw std::runtime_error("Supertonic unicode indexer is empty");
    }
    return out;
}

// The studios install the int8 Supertonic 3 export: the four ONNX graphs of the official release, with the large
// weights stored as int8 and a per-channel scale and zero point for DequantizeLinear. Its initializers keep the
// release names, which the GGML safetensors carry as "<graph>.<name>"; the weights the exporter left anonymous
// (onnx::MatMul_N, onnx::Conv_N, onnx::PRelu_N) take the name of the node that consumes them, in the same layout.
struct OnnxGraph {
    const char * stem;
    const char * scope;
};

constexpr OnnxGraph kOnnxGraphs[] = {
    {"duration_predictor", "tts.dp."},
    {"text_encoder", "tts.ttl."},
    {"vector_estimator", "tts.ttl."},
    {"vocoder", "tts.ae."},
};

bool ends_with(const std::string & value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// "/vector_estimator/vector_field/main_blocks.5/attention/W_value/linear/MatMul", input 1 ->
// "vector_estimator.vector_estimator.tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.weight".
std::string consumer_name(const OnnxGraph & graph, const assets::onnx::Node & node, size_t input) {
    std::vector<std::string> parts;
    std::stringstream path(node.name);
    for (std::string part; std::getline(path, part, '/');) {
        if (!part.empty()) parts.push_back(part);
    }
    if (parts.size() < 2 || input < 1 || input > 2) {
        throw std::runtime_error("Supertonic ONNX has a weight its node does not name: " + node.name);
    }
    parts.pop_back();
    std::string out = std::string(graph.stem) + ".";
    size_t first = 0;
    if (parts.front() == graph.stem) {
        out += std::string(graph.stem) + ".";
        first = 1;
    }
    out += graph.scope;
    for (size_t i = first; i < parts.size(); ++i) out += parts[i] + ".";
    return out + (input == 1 ? "weight" : "bias");
}

std::shared_ptr<const assets::TensorSource> open_onnx_graph(const std::filesystem::path & dir, const OnnxGraph & graph) {
    auto model = std::make_shared<const assets::onnx::Model>(
        assets::onnx::read_model(dir / "onnx" / (std::string(graph.stem) + ".onnx")));
    const auto consumer = [&](const std::string & tensor) -> std::pair<const assets::onnx::Node *, size_t> {
        for (const auto & node : model->nodes) {
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                if (node.inputs[i] == tensor) return {&node, i};
            }
        }
        throw std::runtime_error("Supertonic ONNX has a weight no node reads: " + tensor);
    };
    std::unordered_map<std::string, assets::onnx::Route> routes;
    const auto add = [&](std::string name, assets::onnx::Route route) {
        const std::string source = route.initializer;
        if (!routes.emplace(std::move(name), std::move(route)).second) {
            throw std::runtime_error("Supertonic ONNX names two weights alike, one of them " + source);
        }
    };
    for (const auto & [name, initializer] : model->initializers) {
        const bool quantized = (initializer.dtype == "I8" || initializer.dtype == "U8") && ends_with(name, "_q");
        if (quantized) {
            const auto [dequantize, index] = consumer(name);
            if (dequantize->op_type != "DequantizeLinear" || index != 0 || dequantize->inputs.size() < 2) {
                throw std::runtime_error("Supertonic ONNX int8 weight is not read by DequantizeLinear: " + name);
            }
            assets::onnx::Route route{name, false, dequantize->inputs[1],
                                      dequantize->inputs.size() > 2 ? dequantize->inputs[2] : std::string(), 1};
            if (const auto axis = dequantize->int_attributes.find("axis"); axis != dequantize->int_attributes.end()) {
                route.axis = axis->second;
            }
            const std::string base = name.substr(0, name.size() - 2);
            if (base.rfind("onnx::", 0) == 0) {
                const auto [user, input] = consumer(dequantize->outputs.at(0));
                add(consumer_name(graph, *user, input), std::move(route));
            } else {
                add(std::string(graph.stem) + "." + base, std::move(route));
            }
        } else if ((ends_with(name, "_scale") || ends_with(name, "_zp")) &&
                   model->initializers.count(name.substr(0, name.rfind('_')) + "_q") != 0) {
            continue;
        } else if (name.rfind("onnx::", 0) == 0) {
            if (initializer.dtype != "F32") continue;
            const auto [user, input] = consumer(name);
            add(consumer_name(graph, *user, input), {name});
        } else if (name.front() == '/') {
            std::string logical = name.substr(1);
            std::replace(logical.begin(), logical.end(), '/', '.');
            add(std::string(graph.stem) + "." + logical, {name});
        } else {
            add(std::string(graph.stem) + "." + name, {name});
        }
    }
    return assets::onnx::open_routed(std::move(model), std::move(routes), {});
}

std::shared_ptr<const SupertonicAssets> load_onnx_assets(const std::filesystem::path & dir) {
    auto assets = std::make_shared<SupertonicAssets>();
    assets->resources = assets::ResourceBundle(dir);
    assets->resources.add_model_file("tts_config", "onnx/tts.json");
    assets->resources.add_model_file("unicode_indexer", "onnx/unicode_indexer.json");
    for (const auto & entry : std::filesystem::directory_iterator(dir / "voice_styles")) {
        if (entry.path().extension() == ".json") {
            assets->resources.add_model_file(
                "voice_style_" + entry.path().stem().string(), std::filesystem::path("voice_styles") / entry.path().filename());
        }
    }
    assets->config = parse_config(assets->resources);
    std::vector<std::shared_ptr<const assets::TensorSource>> graphs;
    for (const auto & graph : kOnnxGraphs) graphs.push_back(open_onnx_graph(dir, graph));
    assets->weights = assets::onnx::join(std::move(graphs));
    assets->unicode_indexer = parse_unicode_indexer(assets->resources);
    return assets;
}

}  // namespace

bool is_onnx_directory(const std::filesystem::path & model_path) {
    std::error_code error;
    return std::filesystem::is_regular_file(model_path / "onnx" / "vector_estimator.onnx", error) &&
           std::filesystem::is_regular_file(model_path / "onnx" / "tts.json", error);
}

std::shared_ptr<const SupertonicAssets> load_supertonic_assets(const std::filesystem::path & model_path) {
    if (is_onnx_directory(model_path)) return load_onnx_assets(model_path);
    auto assets = std::make_shared<SupertonicAssets>();
    assets->resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("supertonic"));
    assets->config = parse_config(assets->resources);
    assets->weights = assets->resources.open_tensor_source("weights");
    assets->unicode_indexer = parse_unicode_indexer(assets->resources);
    return assets;
}

}  // namespace engine::models::supertonic
