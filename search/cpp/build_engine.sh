#!/usr/bin/env bash
# 全部 C++ の USI エンジン（LibTorch 版）をビルドする。WSL(Ubuntu) / Linux 用。
#   LibTorch は pip で入れた PyTorch に入っているもの（ヘッダとライブラリ）を使う。
#   使い方: bash search/cpp/build_engine.sh   → search/cpp/apple_engine ができる
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CS="${HERE}/third_party/cshogi/src"
PY="${PYTHON:-python3}"
TORCH_DIR="$(${PY} -c 'import torch, os; print(os.path.dirname(torch.__file__))')"
ABI="$(${PY} -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
ARCH_FLAGS="-msse4.2 -mavx2 -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_AVX2"
if [[ "${NO_AVX2:-0}" == "1" ]]; then ARCH_FLAGS="-msse4.2 -DHAVE_SSE4 -DHAVE_SSE42"; fi

# cshogi 部分は build.sh と共通のオブジェクトを使う
bash "${HERE}/build.sh" >/dev/null

OBJS=()
for s in bitboard common generateMoves hand init move mt64bit position search square usi dfpn; do
  OBJS+=("${HERE}/build/${s}.o")
done

echo "compile engine_main.cpp（少し時間がかかります）"
g++ -O3 -std=c++20 ${ARCH_FLAGS} -w -D_GLIBCXX_USE_CXX11_ABI=${ABI} \
  -I"${CS}" -I"${HERE}/src" \
  -I"${TORCH_DIR}/include" -I"${TORCH_DIR}/include/torch/csrc/api/include" \
  "${HERE}/src/engine_main.cpp" "${OBJS[@]}" \
  -L"${TORCH_DIR}/lib" -Wl,-rpath,"${TORCH_DIR}/lib" \
  -ltorch -ltorch_cpu -lc10 -lpthread \
  -o "${HERE}/apple_engine"
echo "OK: ${HERE}/apple_engine"
