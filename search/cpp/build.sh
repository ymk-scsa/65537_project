#!/usr/bin/env bash
# C++ 探索モジュール（apple_search_cpp）をビルドする。WSL(Ubuntu) / Linux 用。
#   必要: g++ (C++17), python3-dev, pybind11 (pip install pybind11)
#   使い方: bash search/cpp/build.sh   → search/cpp/ に apple_search_cpp*.so ができる
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CS="${HERE}/third_party/cshogi/src"
PY="${PYTHON:-python3}"
EXT="$(${PY} -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
INC="$(${PY} -m pybind11 --includes)"
ARCH_FLAGS="-msse4.2 -mavx2 -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_AVX2"
if [[ "${NO_AVX2:-0}" == "1" ]]; then ARCH_FLAGS="-msse4.2 -DHAVE_SSE4 -DHAVE_SSE42"; fi
SRCS=(bitboard common generateMoves hand init move mt64bit position search square usi dfpn)
OBJDIR="${HERE}/build"
mkdir -p "${OBJDIR}"
OBJS=()
PIDS=()
for s in "${SRCS[@]}"; do
  o="${OBJDIR}/${s}.o"
  if [[ ! -f "${o}" || "${CS}/${s}.cpp" -nt "${o}" ]]; then
    echo "compile ${s}.cpp"
    g++ -O3 -std=c++17 -fPIC ${ARCH_FLAGS} -Wno-enum-constexpr-conversion -w -I"${CS}" -c "${CS}/${s}.cpp" -o "${o}" &
    PIDS+=($!)
  fi
  OBJS+=("${o}")
done
for p in "${PIDS[@]}"; do wait "$p" || { echo "ビルド失敗"; exit 1; }; done
echo "compile py_apple_search.cpp"
g++ -O3 -std=c++17 -fPIC -shared ${ARCH_FLAGS} -w ${INC} -I"${CS}" -I"${HERE}/src" \
  "${HERE}/src/py_apple_search.cpp" "${OBJS[@]}" -o "${HERE}/apple_search_cpp${EXT}"
echo "OK: ${HERE}/apple_search_cpp${EXT}"
