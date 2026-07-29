# xxHash, header-only. Vendored rather than fetched because the point of it is to be byte-identical to
# the copy inside dxvk-remix (src/util/xxHash/xxhash.h): Remix identifies every asset by an XXH3/XXH64
# value, and an asset hash that disagrees by so much as a version bump silently stops matching
# replacements instead of failing loudly. BSD 2-Clause; the licence header is intact in the source file.
#
# INTERFACE, so nothing is compiled here. Only components/remixrt/assethash.cpp includes the header --
# it is 200KB of macros and must not leak into every translation unit that touches Remix.
add_library(xxHash INTERFACE)
target_include_directories(xxHash SYSTEM INTERFACE include)
