#!/bin/bash
# audiocpp_server for Linux x86-64, built in an Ubuntu 22.04 container (glibc 2.35: runs there and on anything newer)
# from the committed HEAD, with the model specs inside (AUDIOCPP_DEPLOYMENT_BUILD), no native CPU flags and no AVX-512.
#   cuda (default): CUDA 13.3, machine code for sm 75/80/86/89/90/120a and no PTX.
#   cuda12:         CUDA 12.9, machine code for sm 52/60/61/70/75/80/86/89/90/120a and no PTX.
#   vulkan:         Vulkan and the CPU (LunarG Vulkan SDK; the system's Vulkan loader at run time).
#   cpu:            the CPU alone.
# RUNPATH looks beside the program, in ../../models/higgs-engine, where a studio keeps the CUDA 13 runtime, cuBLAS and
# cuFFT, and in ../../models/cuda12-runtime, where it keeps those of CUDA 12.9; libgomp is copied beside the program.
#   scripts/build-server-linux-docker.sh <output folder> [cache folder] [cuda|cuda12|vulkan|cpu]
set -euo pipefail

VULKAN_SDK_VERSION="1.4.357.0"

if [ "${1:-}" = "--inside" ]; then
  backend="$2"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q >/dev/null
  apt-get install -y -q --no-install-recommends git ninja-build python3-pip ca-certificates build-essential curl xz-utils perl patchelf >/dev/null
  pip3 install -q "cmake>=3.28" >/dev/null
  flags=(-DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=OFF)
  case "$backend" in
    cuda) flags=(-DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_CUDA_GRAPHS=ON -DENGINE_ENABLE_VULKAN=OFF -DGGML_CUDA_NCCL=OFF
                 "-DCMAKE_CUDA_ARCHITECTURES=75-real;80-real;86-real;89-real;90-real;120a-real") ;;
    cuda12) flags=(-DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_CUDA_GRAPHS=ON -DENGINE_ENABLE_VULKAN=OFF -DGGML_CUDA_NCCL=OFF
                   "-DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets"
                   "-DCMAKE_CUDA_ARCHITECTURES=52-real;60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;120a-real") ;;
    vulkan)
      sdk=/cache/vulkansdk-$VULKAN_SDK_VERSION
      if [ ! -x "$sdk/x86_64/bin/glslc" ]; then
        echo "[..] Vulkan SDK $VULKAN_SDK_VERSION"
        mkdir -p "$sdk"
        curl -fsSL "https://sdk.lunarg.com/sdk/download/$VULKAN_SDK_VERSION/linux/vulkansdk-linux-x86_64-$VULKAN_SDK_VERSION.tar.xz" | tar -xJ -C "$sdk" --strip-components=1
      fi
      export VULKAN_SDK="$sdk/x86_64" PATH="$sdk/x86_64/bin:$PATH"
      cp -P "$sdk"/x86_64/lib/VulkanLoader/lib/libvulkan.so* /usr/lib/x86_64-linux-gnu/
      flags=(-DENGINE_ENABLE_CUDA=OFF -DENGINE_ENABLE_VULKAN=ON -DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so) ;;
    cpu) ;;
    *) echo "[ERROR] backend $backend: cuda, cuda12, vulkan or cpu"; exit 1 ;;
  esac
  echo "[..] source $(git -c safe.directory=/src -C /src rev-parse --short HEAD), $backend"
  rm -rf /cache/src && mkdir -p /cache/src
  git -c safe.directory=/src -C /src archive HEAD | tar -x -C /cache/src
  cd /cache/src
  # settings come from this script alone, not from a cache an earlier run left
  rm -rf "/cache/build-$backend/CMakeCache.txt"
  cmake -B "/cache/build-$backend" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DENGINE_ENABLE_OPENMP=ON -DGGML_OPENMP=ON -DENGINE_ENABLE_METAL=OFF -DENGINE_ENABLE_NATIVE_CPU=OFF -DENGINE_ENABLE_LLAMAFILE=ON \
    -DENGINE_BUILD_TESTS=OFF -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF -DAUDIOCPP_USE_SYSTEM_OPENSSL=OFF \
    -DAUDIOCPP_BUILD_C_API=ON -DAUDIOCPP_VERSION=dev -DAUDIOCPP_MODEL_SET=full \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF \
    -DGGML_AVX_VNNI=OFF -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    '-DCMAKE_BUILD_RPATH=$ORIGIN;$ORIGIN/../../models/higgs-engine;$ORIGIN/../../models/cuda12-runtime' "${flags[@]}" >/dev/null
  echo "[..] build"
  cmake --build "/cache/build-$backend" --target audiocpp_server -j "$(( $(nproc) * 3 / 4 ))" 2>&1 | tail -3
  exe=$(find "/cache/build-$backend" -name audiocpp_server -type f -perm -u+x | head -1)
  mkdir -p "/out/$backend"
  cp "$exe" "/out/$backend/audiocpp_server"
  cp "$(gcc -print-file-name=libgomp.so.1)" "/out/$backend/"
  # the CUDA toolkit's folder of the build machine must not reach a user's machine
  patchelf --set-rpath '$ORIGIN:$ORIGIN/../../models/higgs-engine:$ORIGIN/../../models/cuda12-runtime' "/out/$backend/audiocpp_server"
  readelf -d "/out/$backend/audiocpp_server" | grep -E 'NEEDED|RUNPATH'
  echo "zmm: $(objdump -d --no-show-raw-insn "/out/$backend/audiocpp_server" | grep -c '%zmm' || true)"
  sha256sum "/out/$backend/audiocpp_server"
  echo "[OK] audiocpp_server ($backend)"
  exit 0
fi

out="${1:?output folder}"
cache="${2:-$HOME/audiocpp-linux2204-cache}"
backend="${3:-cuda}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$out" "$cache"
cp "$repo/scripts/build-server-linux-docker.sh" "$out/.build-server-linux-docker.sh"
image=ubuntu:22.04
[ "$backend" = "cuda" ] && image=nvidia/cuda:13.3.1-devel-ubuntu22.04
[ "$backend" = "cuda12" ] && image=nvidia/cuda:12.9.1-devel-ubuntu22.04
docker run --rm --cpus "$(( $(nproc) * 2 / 3 ))" -v "$repo:/src:ro" -v "$(cd "$out" && pwd):/out" -v "$cache/$backend:/cache" "$image" bash /out/.build-server-linux-docker.sh --inside "$backend"
