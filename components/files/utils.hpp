#ifndef COMPONENTS_FILES_UTILS_H
#define COMPONENTS_FILES_UTILS_H

#include <cerrno>
#include <cstddef>
#include <format>
#include <istream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace Files
{
    inline std::streamsize getStreamSizeLeft(std::istream& stream)
    {
        const auto begin = stream.tellg();
        if (stream.fail())
            throw std::runtime_error(
                std::format("Failed to get current file position: {}", std::generic_category().message(errno)));

        stream.seekg(0, std::ios_base::end);
        if (stream.fail())
            throw std::runtime_error(
                std::format("Failed to seek end file position: {}", std::generic_category().message(errno)));

        const auto end = stream.tellg();
        if (stream.fail())
            throw std::runtime_error(
                std::format("Failed to get current file position: {}", std::generic_category().message(errno)));

        stream.seekg(begin);
        if (stream.fail())
            throw std::runtime_error(
                std::format("Failed to seek original file position: {}", std::generic_category().message(errno)));

        return end - begin;
    }

    /// Reads whatever is left of \a stream into a string, in blocks.
    ///
    /// Preferred over std::string(std::istreambuf_iterator<char>(stream), {}), which pulls one character at a
    /// time through the streambuf's virtual interface and regrows the string as it goes. That construct was
    /// the innermost frame of a measured stall while OpenMW's Lua scripts were being loaded, and it appears
    /// in several other places that read a whole file.
    ///
    /// Being honest about the size of the win: the per-character dispatch is not the whole of that stall --
    /// a file open and a size query per script, and the Lua compile itself, are in there too. This removes
    /// one of the three.
    inline std::string readAll(std::istream& stream)
    {
        std::string content;

        // Reserved up front when the stream can say how much is left, so the string is allocated once.
        // Not every stream can answer -- a compressed archive entry need not be seekable -- and that is not
        // an error, it just means the blocks below do the growing instead.
        try
        {
            const std::streamsize left = getStreamSizeLeft(stream);
            if (left > 0)
                content.reserve(static_cast<std::size_t>(left));
        }
        catch (const std::exception&)
        {
            stream.clear();
        }

        constexpr std::streamsize blockSize = 16 * 1024;
        char block[blockSize];

        // read() sets failbit on the final short block, which is why gcount decides rather than the stream
        // state. Once failbit is set a further read is a no-op and gcount is zero, so this terminates.
        while (stream.read(block, blockSize), stream.gcount() > 0)
            content.append(block, static_cast<std::size_t>(stream.gcount()));

        return content;
    }
}

#endif
