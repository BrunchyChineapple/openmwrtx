/// Program to test .nif files both on the FileSystem and in BSA archives.

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include <components/bgsm/file.hpp>
#include <components/files/configurationmanager.hpp>
#include <components/files/constrainedfilestream.hpp>
#include <components/files/conversion.hpp>
#include <components/misc/strings/algorithm.hpp>
#include <components/nif/controller.hpp>
#include <components/nif/data.hpp>
#include <components/nif/niffile.hpp>
#include <components/nif/node.hpp>
#include <components/nif/property.hpp>
#include <components/nif/texture.hpp>
#include <components/vfs/archive.hpp>
#include <components/vfs/bsaarchive.hpp>
#include <components/vfs/filesystemarchive.hpp>
#include <components/vfs/manager.hpp>
#include <components/vfs/recursivedirectoryiterator.hpp>

#include <boost/program_options.hpp>

// Create local aliases for brevity
namespace bpo = boost::program_options;

enum class FileType
{
    BSA,
    BA2,
    BGEM,
    BGSM,
    NIF,
    KF,
    BTO,
    BTR,
    RDT,
    PSA,
    Unknown,
};

enum class FileClass
{
    Archive,
    Material,
    NIF,
    Unknown,
};

std::pair<FileType, FileClass> classifyFile(const std::filesystem::path& filename)
{
    const std::string extension = Misc::StringUtils::lowerCase(Files::pathToUnicodeString(filename.extension()));
    if (extension == ".bsa")
        return { FileType::BSA, FileClass::Archive };
    if (extension == ".ba2")
        return { FileType::BA2, FileClass::Archive };
    if (extension == ".bgem")
        return { FileType::BGEM, FileClass::Material };
    if (extension == ".bgsm")
        return { FileType::BGSM, FileClass::Material };
    if (extension == ".nif")
        return { FileType::NIF, FileClass::NIF };
    if (extension == ".kf")
        return { FileType::KF, FileClass::NIF };
    if (extension == ".bto")
        return { FileType::BTO, FileClass::NIF };
    if (extension == ".btr")
        return { FileType::BTR, FileClass::NIF };
    if (extension == ".rdt")
        return { FileType::RDT, FileClass::NIF };
    if (extension == ".psa")
        return { FileType::PSA, FileClass::NIF };

    return { FileType::Unknown, FileClass::Unknown };
}

std::string getFileTypeName(FileType fileType)
{
    switch (fileType)
    {
        case FileType::BSA:
            return "BSA";
        case FileType::BA2:
            return "BA2";
        case FileType::BGEM:
            return "BGEM";
        case FileType::BGSM:
            return "BGSM";
        case FileType::NIF:
            return "NIF";
        case FileType::KF:
            return "KF";
        case FileType::BTO:
            return "BTO";
        case FileType::BTR:
            return "BTR";
        case FileType::RDT:
            return "RDT";
        case FileType::PSA:
            return "PSA";
        case FileType::Unknown:
        default:
            return {};
    }
}

bool isBSA(const std::filesystem::path& path)
{
    return classifyFile(path).second == FileClass::Archive;
}

/// Structure dump, for asking what an asset actually says rather than inferring it.
///
/// Added because there was no way to read a NIF's texture stages offline. Two attempts at parsing the
/// format by hand both failed silently -- version 4.0.0.2 has no block size table, so a sequential walk has
/// to know the body layout of every record type or it desynchronises and produces plausible nonsense, and
/// a ref in that format indexes the global block list rather than the NiSourceTexture records. Both
/// mistakes cost a session between them.
///
/// This reuses the engine's own reader instead, which removes the entire class of error: if OpenMW can
/// load the file, this reports what OpenMW loaded. That also makes it the right instrument by this
/// project's usual rule, since OpenMW's loader is the consumer whose behaviour is in question.
///
/// The immediate question was where the Dahrk window lattice lives. The host consumes four texture roles
/// and deliberately ignores dark, detail, gloss, bump and decal, so a texture on one of those is bound in
/// the asset, drawn by the rasteriser, and invisible to Remix -- and because nothing consumes it, never
/// even uploaded, so no log line mentions it at all. Absence is the symptom, which is exactly the case
/// where the asset has to be read directly.
namespace NifDump
{
    /// NiTexturingProperty slot order, matching NiTexturingProperty::TextureType and the uniform names
    /// nifloader binds them to.
    const char* slotName(std::size_t slot)
    {
        switch (slot)
        {
            case Nif::NiTexturingProperty::BaseTexture:
                return "base   -> diffuseMap";
            case Nif::NiTexturingProperty::DarkTexture:
                return "dark   -> darkMap";
            case Nif::NiTexturingProperty::DetailTexture:
                return "detail -> detailMap";
            case Nif::NiTexturingProperty::GlossTexture:
                return "gloss  -> glossMap";
            case Nif::NiTexturingProperty::GlowTexture:
                return "glow   -> emissiveMap";
            case Nif::NiTexturingProperty::BumpTexture:
                return "bump   -> bumpMap";
            case Nif::NiTexturingProperty::DecalTexture:
                return "decal  -> decalMap";
            default:
                return "extra decal";
        }
    }

    const char* applyModeName(Nif::NiTexturingProperty::ApplyMode mode)
    {
        switch (mode)
        {
            case Nif::NiTexturingProperty::ApplyMode::Replace:
                return "Replace";
            case Nif::NiTexturingProperty::ApplyMode::Decal:
                return "Decal";
            case Nif::NiTexturingProperty::ApplyMode::Modulate:
                return "Modulate";
            case Nif::NiTexturingProperty::ApplyMode::Hilight:
                return "Hilight";
            case Nif::NiTexturingProperty::ApplyMode::Hilight2:
                return "Hilight2";
            default:
                return "?";
        }
    }

    std::string sourceFile(const Nif::NiSourceTexture* source)
    {
        if (source == nullptr)
            return "(null source)";
        if (!source->mExternal)
            return "(internal pixel data)";
        return source->mFile;
    }

    bool matchesFilter(const Nif::NiTexturingProperty& prop, const std::string& filter)
    {
        if (filter.empty())
            return true;
        for (const Nif::NiTexturingProperty::Texture& texture : prop.mTextures)
        {
            if (!texture.mEnabled)
                continue;
            if (Misc::StringUtils::ciFind(sourceFile(texture.mSourceTexture.getPtr()), filter)
                != std::string::npos)
                return true;
        }
        return false;
    }

    void dumpTexturing(const Nif::NiTexturingProperty& prop, unsigned int index)
    {
        std::cout << "  [" << index << "] NiTexturingProperty name='" << prop.mName << "' applyMode="
                  << applyModeName(prop.mApplyMode) << " flags=0x" << std::hex << prop.mFlags << std::dec
                  << " slots=" << prop.mTextures.size() << '\n';
        bool hasBase = false;
        for (std::size_t slot = 0; slot < prop.mTextures.size(); ++slot)
        {
            const Nif::NiTexturingProperty::Texture& texture = prop.mTextures[slot];
            if (!texture.mEnabled)
                continue;
            if (slot == Nif::NiTexturingProperty::BaseTexture)
                hasBase = true;
            std::cout << "        " << slotName(slot) << "  uvSet=" << texture.mUVSet
                      << " clamp=" << texture.mClamp << " filter=" << texture.mFilter << "  "
                      << sourceFile(texture.mSourceTexture.getPtr()) << '\n';
            // The transform is why an earlier attempt at forcing one of these stages through came out
            // flipped and zoomed: a stage carries its own offset, scale, rotation and centre, and sampling
            // it with the base stage's texcoords ignores all four.
            if (texture.mHasTransform)
            {
                const Nif::NiTextureTransform& t = texture.mTransform;
                std::cout << "            transform offset=(" << t.mOffset.x() << ", " << t.mOffset.y()
                          << ") scale=(" << t.mScale.x() << ", " << t.mScale.y() << ")\n";
            }
        }
        if (!hasBase)
            std::cout << "        NO BASE SLOT -- nifloader binds no diffuseMap, so the host assigns no "
                         "albedo and the surface falls back to its default material\n";
    }

    void dumpMaterial(const Nif::NiMaterialProperty& prop, unsigned int index)
    {
        std::cout << "  [" << index << "] NiMaterialProperty name='" << prop.mName << "'"
                  << " diffuse=(" << prop.mDiffuse.x() << ", " << prop.mDiffuse.y() << ", "
                  << prop.mDiffuse.z() << ")"
                  << " emissive=(" << prop.mEmissive.x() << ", " << prop.mEmissive.y() << ", "
                  << prop.mEmissive.z() << ")"
                  << " emissiveMult=" << prop.mEmissiveMult << " alpha=" << prop.mAlpha
                  << " glossiness=" << prop.mGlossiness << '\n';
    }

    void dumpAlpha(const Nif::NiAlphaProperty& prop, unsigned int index)
    {
        std::cout << "  [" << index << "] NiAlphaProperty name='" << prop.mName << "' flags=0x" << std::hex
                  << prop.mFlags << std::dec << " threshold=" << static_cast<unsigned int>(prop.mThreshold)
                  << " blending=" << (prop.useAlphaBlending() ? "yes" : "no")
                  << " testing=" << (prop.useAlphaTesting() ? "yes" : "no") << '\n';
    }

    /// One line of the node tree, with each node's own properties named.
    ///
    /// The flat record listing says which texture stages exist; it cannot say which of them the engine
    /// will actually show. For a Glow in the Dahrk window that is the whole question: the mesh holds a day
    /// variant, a night variant and sometimes a third, and only the NiSwitchNode decides. So the tree is
    /// printed with each switch node's initial index and each shape's property indices, which is what
    /// connects a texturing property in the listing above to a branch that is or is not selected.
    void dumpTree(const Nif::Record* record, unsigned int depth, unsigned int childIndex, bool underSwitch)
    {
        if (record == nullptr || depth > 12)
            return;

        const auto* object = dynamic_cast<const Nif::NiAVObject*>(record);
        std::string indent(static_cast<std::size_t>(depth) * 4, ' ');

        std::cout << "      " << indent;
        if (underSwitch)
            std::cout << "child " << childIndex << ": ";
        std::cout << record->mRecordName;
        if (object != nullptr && !object->mName.empty())
            std::cout << " '" << object->mName << "'";
        std::cout << " [" << record->mRecordIndex << "]";

        if (const auto* switchNode = dynamic_cast<const Nif::NiSwitchNode*>(record))
            std::cout << "  SWITCH initialIndex=" << switchNode->mInitialIndex << " flags=0x" << std::hex
                      << switchNode->mSwitchFlags << std::dec;

        if (object != nullptr && !object->mProperties.empty())
        {
            std::cout << "  properties:";
            for (const auto& property : object->mProperties)
            {
                const Nif::Record* rec = property.getPtr();
                if (rec == nullptr)
                    continue;
                std::cout << " " << rec->mRecordName << "[" << rec->mRecordIndex << "]";
            }
        }

        // The controller chain, and for a NiUVController which UV set it drives.
        //
        // That field decides whether a combination of texture stages can be precomputed. If the animation
        // moves the set the base stage samples, a static combination stays valid and the motion rides on
        // top as an ordinary texture matrix. If it moves the set an ignored stage samples, a static
        // combination is wrong every frame after the first, and nothing about the result would say so.
        if (const auto* net = dynamic_cast<const Nif::NiObjectNET*>(record))
        {
            const Nif::NiTimeController* controller = net->mController.getPtr();
            while (controller != nullptr)
            {
                std::cout << "  controller:" << controller->mRecordName;
                if (const auto* uv = dynamic_cast<const Nif::NiUVController*>(controller))
                    std::cout << "(animates uvSet " << uv->mUvSet << ")";
                controller = controller->mNext.getPtr();
            }
        }
        std::cout << '\n';

        // The UV sets, and whether they actually differ.
        //
        // A texture stage names a uvSet index, and whether combining several stages is easy or impossible
        // turns entirely on this. Different coordinates per set means the correspondence between stages is
        // per-vertex geometry, which cannot be expressed in texture space -- that is why flattening stages
        // into one image was tried on this project and abandoned. Identical sets mean the indices are
        // decoration and every stage can be sampled with one set of texcoords, which makes the same
        // combination a per-texel multiply.
        //
        // Compared exactly rather than approximately: these are duplicated on export or they are authored
        // apart, so there is no near-miss case worth a tolerance.
        if (const auto* geometry = dynamic_cast<const Nif::NiGeometry*>(record))
        {
            if (const Nif::NiGeometryData* data = geometry->mData.getPtr())
            {
                std::cout << "      " << indent << "    data [" << data->mRecordIndex << "] vertices "
                          << data->mVertices.size() << ", uvSets " << data->mUVList.size();
                for (std::size_t set = 1; set < data->mUVList.size(); ++set)
                {
                    const bool same = data->mUVList[set] == data->mUVList[0];
                    std::cout << "; set " << set << (same ? " IDENTICAL to set 0" : " DIFFERS from set 0");
                    if (!same && data->mUVList[set].size() == data->mUVList[0].size()
                        && !data->mUVList[set].empty())
                    {
                        // Deltas alone cannot distinguish "remapped per vertex" from "scaled", because a
                        // scale produces deltas proportional to position. That distinction is the whole
                        // question -- an affine relationship is expressible as a texture transform and a
                        // per-vertex one is not -- so on a small shape the coordinates are printed and the
                        // relationship can be read directly instead of inferred from a range.
                        if (data->mUVList[0].size() <= 32)
                        {
                            std::cout << "\n      " << indent << "      set 0 vs set " << set << ":";
                            for (std::size_t i = 0; i < data->mUVList[0].size(); ++i)
                                std::cout << "\n      " << indent << "        [" << i << "] ("
                                          << data->mUVList[0][i].x() << ", " << data->mUVList[0][i].y()
                                          << ") -> (" << data->mUVList[set][i].x() << ", "
                                          << data->mUVList[set][i].y() << ")";
                            std::cout << "\n      " << indent << "     ";
                        }
                        // How far apart, so "differs" can be told from "differs by a constant offset",
                        // which a single texture matrix could still absorb.
                        osg::Vec2f minDelta(1e9f, 1e9f);
                        osg::Vec2f maxDelta(-1e9f, -1e9f);
                        for (std::size_t i = 0; i < data->mUVList[set].size(); ++i)
                        {
                            const osg::Vec2f delta = data->mUVList[set][i] - data->mUVList[0][i];
                            minDelta.x() = std::min(minDelta.x(), delta.x());
                            minDelta.y() = std::min(minDelta.y(), delta.y());
                            maxDelta.x() = std::max(maxDelta.x(), delta.x());
                            maxDelta.y() = std::max(maxDelta.y(), delta.y());
                        }
                        std::cout << " (delta from set 0 ranges x " << minDelta.x() << ".." << maxDelta.x()
                                  << ", y " << minDelta.y() << ".." << maxDelta.y() << ")";
                    }
                }
                std::cout << '\n';
            }
        }

        if (const auto* node = dynamic_cast<const Nif::NiNode*>(record))
        {
            const bool isSwitch = dynamic_cast<const Nif::NiSwitchNode*>(record) != nullptr;
            unsigned int index = 0;
            for (const auto& child : node->mChildren)
            {
                dumpTree(child.getPtr(), depth + 1, index, isSwitch);
                ++index;
            }
        }
    }

    void dump(const Nif::NIFFile& file, const std::string& filter)
    {
        std::cout << "==== " << file.mPath << '\n';
        std::cout << "  version=0x" << std::hex << file.mVersion << std::dec
                  << " userVersion=" << file.mUserVersion << " bethVersion=" << file.mBethVersion
                  << " records=" << file.mRecords.size() << " roots=" << file.mRoots.size() << '\n';

        std::map<std::string, unsigned int> histogram;
        for (const std::unique_ptr<Nif::Record>& record : file.mRecords)
            if (record != nullptr)
                ++histogram[record->mRecordName];

        std::cout << "  record types:\n";
        for (const auto& [name, count] : histogram)
            std::cout << "      " << name << " x" << count << '\n';

        if (!filter.empty())
            std::cout << "  filtering texturing properties to those referencing '" << filter << "'\n";

        unsigned int shown = 0;
        for (unsigned int i = 0; i < file.mRecords.size(); ++i)
        {
            const Nif::Record* record = file.mRecords[i].get();
            if (record == nullptr)
                continue;
            switch (record->mRecordType)
            {
                case Nif::RC_NiTexturingProperty:
                {
                    const auto& prop = static_cast<const Nif::NiTexturingProperty&>(*record);
                    if (!matchesFilter(prop, filter))
                        break;
                    dumpTexturing(prop, i);
                    ++shown;
                    break;
                }
                // Printed unfiltered: there are only ever a handful, and a material's emissive is how
                // Morrowind states that a whole surface glows, which is the other half of this question.
                case Nif::RC_NiMaterialProperty:
                    dumpMaterial(static_cast<const Nif::NiMaterialProperty&>(*record), i);
                    ++shown;
                    break;
                case Nif::RC_NiAlphaProperty:
                    dumpAlpha(static_cast<const Nif::NiAlphaProperty&>(*record), i);
                    ++shown;
                    break;
                default:
                    break;
            }
        }
        if (shown == 0)
            std::cout << "  nothing matched\n";

        // Only for files small enough to read. A merged town is 7,445 nodes and the tree is noise there;
        // a single window asset is a dozen lines and the tree is the entire point.
        if (file.mRecords.size() <= 400)
        {
            std::cout << "  node tree:\n";
            for (const Nif::Record* root : file.mRoots)
                dumpTree(root, 0, 0, false);
        }
        else
        {
            std::cout << "  node tree omitted (" << file.mRecords.size()
                      << " records; --dump prints it below 400)\n";
        }
        std::cout << std::endl;
    }
}

/// Aggregate survey of every NIF in a data set, reported once at the end rather than per file.
///
/// --dump answers "what does this one mesh declare". This answers "what does the installed asset set contain
/// that the host has to cope with", which is a different question and cannot be answered by reading dumps:
/// there are tens of thousands of meshes, and the features that matter are rare by proportion and
/// significant by consequence. A stage on a second uv set is 3.8% of texturing properties and was the entire
/// Glow in the Dahrk defect.
///
/// The classification is deliberately phrased as what the *host* will do, not as what the format contains,
/// so the output is a work list. A count under "must fold, animation freezes" is a count of surfaces that
/// will be visibly wrong until the phase sprite sheet exists.
namespace NifCensus
{
    /// A count plus a few names, because a bare number invites the next question and a full list is useless.
    struct Bucket
    {
        unsigned int mCount = 0;
        std::vector<std::string> mExamples;

        void note(const std::string& file)
        {
            ++mCount;
            if (mExamples.size() < 8)
            {
                // Deduplicated: one mesh can contribute many shapes to the same bucket, and eight copies of
                // one filename is a worse sample than one each of eight files.
                if (std::find(mExamples.begin(), mExamples.end(), file) == mExamples.end())
                    mExamples.push_back(file);
            }
        }
    };

    struct Stats
    {
        unsigned int mFiles = 0;
        unsigned int mFailed = 0;
        std::map<std::string, unsigned int> mVersions;
        std::map<std::string, unsigned int> mRecords;
        std::map<std::string, unsigned int> mRecordFiles;
        std::map<std::string, Bucket> mSlotCombos;
        std::map<std::string, unsigned int> mApplyModes;
        std::map<std::string, unsigned int> mTextureExtensions;
        std::set<std::string> mTextures;

        // Triangles per file and the total, for the renderer's scene-wide primitive limits. Kept as a flat
        // list and sorted at report time rather than maintained as a heap: 60k entries is nothing next to
        // parsing the files themselves, and the whole list is wanted for the total anyway.
        std::vector<std::pair<std::size_t, std::string>> mTrianglesByFile;
        std::size_t mTrianglesTotal = 0;

        unsigned int mShapes = 0;
        unsigned int mShapesUntextured = 0;
        unsigned int mShapesNoBaseSlot = 0;
        unsigned int mShapesMultiStage = 0;
        unsigned int mShapesAnimatedUv = 0;

        Bucket mFactorable;
        Bucket mMustFold;
        Bucket mIdenticalUvTrap;
        Bucket mAnimatedMustFold;
        Bucket mStageTransform;
        Bucket mBaseOffAnimatedSet;
        Bucket mOneAffineFits;
        Bucket mPerTriangleRequired;
        float mWorstFittingResidual = 0.f;

        unsigned int mMaterials = 0;
        unsigned int mMaterialsEmissive = 0;
        unsigned int mMaterialsAlphaBelowOne = 0;
        Bucket mBlackDiffuseEmissive;

        std::map<std::string, unsigned int> mBlendModes;
        unsigned int mAlphaTested = 0;
        unsigned int mAlphaBlended = 0;

        Bucket mTextureEffects;
        Bucket mParticleSystems;
        Bucket mSkinned;
        Bucket mSwitchNodes;
        Bucket mMultiUvGeometry;
    };

    Stats gStats;

    const char* shortSlotName(std::size_t slot)
    {
        switch (slot)
        {
            case Nif::NiTexturingProperty::BaseTexture:
                return "base";
            case Nif::NiTexturingProperty::DarkTexture:
                return "dark";
            case Nif::NiTexturingProperty::DetailTexture:
                return "detail";
            case Nif::NiTexturingProperty::GlossTexture:
                return "gloss";
            case Nif::NiTexturingProperty::GlowTexture:
                return "glow";
            case Nif::NiTexturingProperty::BumpTexture:
                return "bump";
            case Nif::NiTexturingProperty::DecalTexture:
                return "decal";
            default:
                return "extra";
        }
    }

    /// Blend factors, named, so additive can be told from ordinary transparency in the report.
    ///
    /// The distinction is not cosmetic: the host turns a destination factor of ONE into an emissive material
    /// and everything else into either a cutout or real blending, and those are three different appearances.
    const char* blendFactorName(unsigned int factor)
    {
        switch (factor)
        {
            case 0:
                return "ONE";
            case 1:
                return "ZERO";
            case 2:
                return "SRC_COLOR";
            case 3:
                return "INV_SRC_COLOR";
            case 4:
                return "DST_COLOR";
            case 5:
                return "INV_DST_COLOR";
            case 6:
                return "SRC_ALPHA";
            case 7:
                return "INV_SRC_ALPHA";
            case 8:
                return "DST_ALPHA";
            case 9:
                return "INV_DST_ALPHA";
            case 10:
                return "SRC_ALPHA_SAT";
            default:
                return "?";
        }
    }

    /// Whether a single affine maps every vertex's base UV to its stage UV, and how badly it fails if not.
    ///
    /// This is the question that decides whether the stage combination needs per-triangle work at all. A
    /// per-vertex UV mapping is piecewise linear, so it is affine *within* every triangle by construction --
    /// but if one affine happens to describe the whole surface, the relationship can be expressed as a texture
    /// transform and handed to a shader, which is far cheaper than baking an image. Two earlier attempts on
    /// this project assumed one affine always fits, and were defeated by the meshes where it does not.
    ///
    /// Three non-collinear points determine an affine, so this solves from the widest triangle it can find in
    /// base UV space and then verifies every vertex against it. No topology needed. The widest is chosen
    /// deliberately: solving from an almost-collinear triple gives an affine that fits those three points and
    /// nothing else, which would report a false failure on a surface that one affine does describe.
    bool oneAffineFits(const std::vector<osg::Vec2f>& base, const std::vector<osg::Vec2f>& stage,
        float& outResidual)
    {
        outResidual = 0.f;
        if (base.size() < 3 || base.size() != stage.size())
            return false;

        // Widest triangle: the two extremes along the first axis, plus whichever point is furthest from the
        // line between them.
        std::size_t a = 0;
        std::size_t b = 0;
        for (std::size_t i = 1; i < base.size(); ++i)
        {
            if (base[i].x() < base[a].x() || (base[i].x() == base[a].x() && base[i].y() < base[a].y()))
                a = i;
            if (base[i].x() > base[b].x() || (base[i].x() == base[b].x() && base[i].y() > base[b].y()))
                b = i;
        }
        if (a == b)
            return false;

        std::size_t c = a;
        double best = 0.0;
        for (std::size_t i = 0; i < base.size(); ++i)
        {
            const double cross = static_cast<double>(base[b].x() - base[a].x()) * (base[i].y() - base[a].y())
                - static_cast<double>(base[b].y() - base[a].y()) * (base[i].x() - base[a].x());
            if (std::abs(cross) > best)
            {
                best = std::abs(cross);
                c = i;
            }
        }

        // Degenerate in base UV space: every vertex on one line, so no affine is determined and there is
        // nothing to classify. Reported as "does not fit" rather than silently counted as fitting.
        const double det = static_cast<double>(base[b].x() - base[a].x()) * (base[c].y() - base[a].y())
            - static_cast<double>(base[b].y() - base[a].y()) * (base[c].x() - base[a].x());
        if (std::abs(det) < 1e-12)
            return false;

        // Solve stage = M * base + t from the three points, one output axis at a time.
        const auto solveAxis = [&](float sa, float sb, float sc, double& m0, double& m1, double& t) {
            const double u0 = base[a].x(), v0 = base[a].y();
            const double u1 = base[b].x(), v1 = base[b].y();
            const double u2 = base[c].x(), v2 = base[c].y();
            m0 = ((static_cast<double>(sb) - sa) * (v2 - v0) - (static_cast<double>(sc) - sa) * (v1 - v0)) / det;
            m1 = ((static_cast<double>(sc) - sa) * (u1 - u0) - (static_cast<double>(sb) - sa) * (u2 - u0)) / det;
            t = sa - m0 * u0 - m1 * v0;
        };
        double xm0, xm1, xt, ym0, ym1, yt;
        solveAxis(stage[a].x(), stage[b].x(), stage[c].x(), xm0, xm1, xt);
        solveAxis(stage[a].y(), stage[b].y(), stage[c].y(), ym0, ym1, yt);

        for (std::size_t i = 0; i < base.size(); ++i)
        {
            const double px = xm0 * base[i].x() + xm1 * base[i].y() + xt;
            const double py = ym0 * base[i].x() + ym1 * base[i].y() + yt;
            outResidual = std::max(outResidual,
                static_cast<float>(std::max(std::abs(px - stage[i].x()), std::abs(py - stage[i].y()))));
        }

        // A UV unit is the whole texture, so a thousandth of one is a third of a texel at 512 across.
        //
        // This threshold *is* somewhat load-bearing, contrary to what a first version of this comment
        // claimed. The worst residual among the shapes it accepts measures 5.8e-4 across the installed set,
        // so there is under a factor of two of headroom rather than the orders of magnitude the clear-cut
        // cases suggested -- a handful of surfaces genuinely sit near the line and would change bucket if it
        // moved. Treat the split it reports as approximate. The report prints the worst accepted residual for
        // exactly this reason.
        return outResidual < 1e-3f;
    }

    /// What a shape inherits from its ancestors. NIF properties and controllers sit on nodes as well as on
    /// shapes, and OpenMW's own loader collects them down the path, so a census that only looked at the
    /// shape's own records would miss whatever a NiBSAnimationNode parent contributes.
    struct Inherited
    {
        const Nif::NiTexturingProperty* mTexturing = nullptr;
        int mAnimatedUvSet = -1;
    };

    void classifyShape(const Nif::NiGeometry& geometry, const Inherited& inherited, const std::string& file)
    {
        ++gStats.mShapes;
        if (inherited.mTexturing == nullptr)
        {
            ++gStats.mShapesUntextured;
            return;
        }

        const Nif::NiTexturingProperty& tex = *inherited.mTexturing;

        std::string combo;
        bool hasBase = false;
        int baseUv = 0;
        // Only dark and detail drive the stage bake. Glow goes to the emissive slot on its own, and the rest
        // are captured and unconsumed, so folding them into this classification would inflate the work list.
        std::vector<std::pair<std::size_t, int>> baked;

        for (std::size_t slot = 0; slot < tex.mTextures.size(); ++slot)
        {
            const Nif::NiTexturingProperty::Texture& texture = tex.mTextures[slot];
            if (!texture.mEnabled)
                continue;

            if (!combo.empty())
                combo += '+';
            combo += shortSlotName(slot);

            if (slot == Nif::NiTexturingProperty::BaseTexture)
            {
                hasBase = true;
                baseUv = texture.mUVSet;
            }
            else if (slot == Nif::NiTexturingProperty::DarkTexture
                || slot == Nif::NiTexturingProperty::DetailTexture)
            {
                baked.emplace_back(slot, texture.mUVSet);
            }

            if (texture.mHasTransform)
                gStats.mStageTransform.note(file);
        }

        if (combo.empty())
            combo = "(no enabled slot)";
        gStats.mSlotCombos[combo].note(file);
        if (!hasBase)
            ++gStats.mShapesNoBaseSlot;

        if (inherited.mAnimatedUvSet >= 0)
        {
            ++gStats.mShapesAnimatedUv;
            // A base that does not read the animated set is worth knowing about separately: the host reads
            // its texture matrix from unit 0, so an animation aimed elsewhere arrives on the wrong stage.
            if (hasBase && baseUv != inherited.mAnimatedUvSet)
                gStats.mBaseOffAnimatedSet.note(file);
        }

        if (baked.empty())
            return;
        ++gStats.mShapesMultiStage;

        const Nif::NiGeometryData* data = geometry.mData.getPtr();

        bool anyOffBaseSet = false;
        bool anyIdenticalContent = false;
        for (const auto& [slot, uv] : baked)
        {
            if (uv == baseUv)
                continue;
            anyOffBaseSet = true;

            // The trap the logs-on-fire mesh sprang. Two uv sets can hold identical coordinates while being
            // different sets, and only the set index decides whether a UV controller drives that unit. A
            // host testing coordinate equality concludes the stage moves with the base when it does not.
            if (data != nullptr && static_cast<std::size_t>(uv) < data->mUVList.size()
                && static_cast<std::size_t>(baseUv) < data->mUVList.size())
            {
                if (data->mUVList[uv] == data->mUVList[baseUv])
                {
                    anyIdenticalContent = true;
                }
                else
                {
                    // Whether this stage's relationship to the base is one affine or per-triangle. Decides
                    // whether the combination could be expressed as a texture transform in a shader instead
                    // of being baked into an image.
                    float residual = 0.f;
                    if (oneAffineFits(data->mUVList[baseUv], data->mUVList[uv], residual))
                    {
                        gStats.mOneAffineFits.note(file);
                        gStats.mWorstFittingResidual = std::max(gStats.mWorstFittingResidual, residual);
                    }
                    else
                    {
                        gStats.mPerTriangleRequired.note(file);
                    }
                }
            }
        }

        if (anyOffBaseSet)
            gStats.mMustFold.note(file);
        else
            gStats.mFactorable.note(file);

        if (anyIdenticalContent)
            gStats.mIdenticalUvTrap.note(file);

        // The count that sizes the phase sprite sheet: animated, and with a baked stage that does not move
        // with the animation, so the bake has to freeze it.
        if (inherited.mAnimatedUvSet >= 0)
        {
            for (const auto& [slot, uv] : baked)
            {
                if (uv != inherited.mAnimatedUvSet)
                {
                    gStats.mAnimatedMustFold.note(file);
                    break;
                }
            }
        }
    }

    void walk(const Nif::Record* record, Inherited inherited, const std::string& file, unsigned int depth)
    {
        if (record == nullptr || depth > 24)
            return;

        // Walked for the uv set a controller animates, and for nothing else. Controllers are not *counted*
        // here: they are records, so the flat histogram already counts each exactly once, whereas a walk
        // counts a shared property's controller once per shape that inherits it. An earlier version of this
        // reported both and disagreed with itself -- 1,201 NiAlphaController records against 0 in the chain,
        // because an alpha controller hangs off the property rather than off a node in the child tree.
        if (const auto* net = dynamic_cast<const Nif::NiObjectNET*>(record))
        {
            for (const Nif::NiTimeController* controller = net->mController.getPtr(); controller != nullptr;
                 controller = controller->mNext.getPtr())
            {
                if (const auto* uv = dynamic_cast<const Nif::NiUVController*>(controller))
                    inherited.mAnimatedUvSet = static_cast<int>(uv->mUvSet);
            }
        }

        if (const auto* object = dynamic_cast<const Nif::NiAVObject*>(record))
        {
            for (const auto& property : object->mProperties)
            {
                const Nif::Record* rec = property.getPtr();
                if (rec == nullptr)
                    continue;
                if (const auto* texturing = dynamic_cast<const Nif::NiTexturingProperty*>(rec))
                    inherited.mTexturing = texturing;
            }
        }

        if (dynamic_cast<const Nif::NiSwitchNode*>(record) != nullptr)
            gStats.mSwitchNodes.note(file);

        if (const auto* geometry = dynamic_cast<const Nif::NiGeometry*>(record))
        {
            // Particles are geometry in the format but never reach the stage bake -- the submit traversal
            // hands them to submitParticles and returns -- so they are counted and then left out of the
            // shape classification rather than inflating it.
            if (dynamic_cast<const Nif::NiParticles*>(record) != nullptr)
                gStats.mParticleSystems.note(file);
            else
                classifyShape(*geometry, inherited, file);

            if (const Nif::NiGeometryData* data = geometry->mData.getPtr())
                if (data->mUVList.size() > 1)
                    gStats.mMultiUvGeometry.note(file);

            if (!geometry->mSkin.empty())
                gStats.mSkinned.note(file);
        }

        if (record->mRecordName == "NiTextureEffect")
            gStats.mTextureEffects.note(file);

        if (const auto* node = dynamic_cast<const Nif::NiNode*>(record))
            for (const auto& child : node->mChildren)
                walk(child.getPtr(), inherited, file, depth + 1);
    }

    void accumulate(const Nif::NIFFile& nif)
    {
        ++gStats.mFiles;
        const std::string file = nif.mPath;

        {
            std::ostringstream version;
            version << "0x" << std::hex << nif.mVersion << std::dec << " user " << nif.mUserVersion
                    << " beth " << nif.mBethVersion;
            ++gStats.mVersions[version.str()];
        }

        // Triangles across every geometry record in the file, summed.
        //
        // Per FILE rather than per shape because that is the unit the renderer pays for: the host merges a
        // file's shapes into one mesh, so one BLAS geometry carries the whole total. NiGeometryData's own
        // mNumTriangles is a uint16_t and so caps at 65,535, which means a report of a million-triangle
        // geometry can only ever be an aggregate -- counting shapes individually hides exactly the asset
        // that matters.
        //
        // Worth having because the runtime enforces hard limits on the sum across the scene: a 26-bit
        // primitive index (67,108,863) and a 24-bit NEE cache prefix sum. Past those, cached light samples
        // resolve to the wrong surface and the index is out of range. The runtime names the offender only by
        // geometry hash, which is not traceable to an asset, so the triangle count is the join column.
        std::size_t trianglesThisFile = 0;

        std::set<std::string> seenHere;
        for (const std::unique_ptr<Nif::Record>& record : nif.mRecords)
        {
            if (record == nullptr)
                continue;
            ++gStats.mRecords[record->mRecordName];
            seenHere.insert(record->mRecordName);

            if (record->mRecordType == Nif::RC_NiTriShapeData)
            {
                trianglesThisFile += static_cast<const Nif::NiTriShapeData&>(*record).mTriangles.size() / 3;
            }
            else if (record->mRecordType == Nif::RC_NiTriStripsData)
            {
                // A strip of n indices is n-2 triangles, degenerate ones included: they are still primitives
                // as far as the acceleration structure and the primitive index are concerned.
                for (const std::vector<unsigned short>& strip :
                    static_cast<const Nif::NiTriStripsData&>(*record).mStrips)
                    if (strip.size() > 2)
                        trianglesThisFile += strip.size() - 2;
            }

            switch (record->mRecordType)
            {
                case Nif::RC_NiTexturingProperty:
                {
                    const auto& prop = static_cast<const Nif::NiTexturingProperty&>(*record);
                    ++gStats.mApplyModes[NifDump::applyModeName(prop.mApplyMode)];
                    for (const Nif::NiTexturingProperty::Texture& texture : prop.mTextures)
                    {
                        if (!texture.mEnabled)
                            continue;
                        const std::string name = Misc::StringUtils::lowerCase(
                            NifDump::sourceFile(texture.mSourceTexture.getPtr()));
                        if (name.empty() || name.front() == '(')
                            continue;
                        gStats.mTextures.insert(name);
                        const std::size_t dot = name.rfind('.');
                        ++gStats.mTextureExtensions[dot == std::string::npos ? "(none)" : name.substr(dot)];
                    }
                    break;
                }
                case Nif::RC_NiMaterialProperty:
                {
                    const auto& prop = static_cast<const Nif::NiMaterialProperty&>(*record);
                    ++gStats.mMaterials;
                    const bool emits = prop.mEmissive.x() > 0.f || prop.mEmissive.y() > 0.f
                        || prop.mEmissive.z() > 0.f;
                    if (emits)
                        ++gStats.mMaterialsEmissive;
                    if (prop.mAlpha < 1.f)
                        ++gStats.mMaterialsAlphaBelowOne;
                    // Purely emissive: black diffuse with a non-black emissive. The rasteriser ADDS the two,
                    // so the surface shows at full brightness; a host that multiplies the diffuse into the
                    // albedo instead renders it black. This is the soul gem spray's material.
                    if (emits && prop.mDiffuse.x() == 0.f && prop.mDiffuse.y() == 0.f
                        && prop.mDiffuse.z() == 0.f)
                    {
                        gStats.mBlackDiffuseEmissive.note(file);
                    }
                    break;
                }
                case Nif::RC_NiAlphaProperty:
                {
                    const auto& prop = static_cast<const Nif::NiAlphaProperty&>(*record);
                    if (prop.useAlphaBlending())
                    {
                        ++gStats.mAlphaBlended;
                        const unsigned int src = (prop.mFlags >> 1) & 0xF;
                        const unsigned int dst = (prop.mFlags >> 5) & 0xF;
                        ++gStats.mBlendModes[std::string(blendFactorName(src)) + " / " + blendFactorName(dst)];
                    }
                    if (prop.useAlphaTesting())
                        ++gStats.mAlphaTested;
                    break;
                }
                default:
                    break;
            }
        }

        for (const std::string& name : seenHere)
            ++gStats.mRecordFiles[name];

        if (trianglesThisFile > 0)
        {
            gStats.mTrianglesByFile.emplace_back(trianglesThisFile, file);
            gStats.mTrianglesTotal += trianglesThisFile;
        }

        for (const Nif::Record* root : nif.mRoots)
            walk(root, Inherited{}, file, 0);
    }

    void noteFailure()
    {
        ++gStats.mFailed;
    }

    void reportBucket(const char* label, const Bucket& bucket, const char* note)
    {
        std::cout << "  " << label << ": " << bucket.mCount << '\n';
        if (note != nullptr && bucket.mCount > 0)
            std::cout << "      " << note << '\n';
        for (const std::string& example : bucket.mExamples)
            std::cout << "        " << example << '\n';
    }

    void report()
    {
        std::cout << "\n================ NIF CENSUS ================\n";
        std::cout << "files parsed: " << gStats.mFiles << ", failed to parse: " << gStats.mFailed << '\n';

        // Heaviest files by triangle count.
        //
        // The renderer caps the whole scene at a 26-bit primitive index, 67,108,863, and the NEE cache at a
        // 24-bit prefix sum. Past either, cached light samples resolve to the wrong surface and the index runs
        // out of range. Cost is per INSTANCE, so an asset's danger is its triangle count times how often it is
        // placed -- a heavy mesh used once is affordable and a heavy mesh used two dozen times is not.
        //
        // The runtime reports offenders by geometry hash only, which no asset can be traced back to, so this
        // list exists to be matched on triangle count instead.
        {
            std::vector<std::pair<std::size_t, std::string>> heaviest = gStats.mTrianglesByFile;
            std::sort(heaviest.begin(), heaviest.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            std::cout << "--- triangles ---\n";
            std::cout << "  total across all parsed files: " << gStats.mTrianglesTotal << '\n';
            std::cout << "  scene primitive index ceiling: 67108863 (26-bit), NEE cache: 16777214 (24-bit)\n";
            std::cout << "  heaviest files (count is per instance; multiply by placements):\n";
            const std::size_t show = std::min<std::size_t>(heaviest.size(), 30);
            for (std::size_t i = 0; i < show; ++i)
                std::cout << "        " << heaviest[i].first << "  " << heaviest[i].second << '\n';
        }

        std::cout << "\n--- NIF versions ---\n";
        for (const auto& [version, count] : gStats.mVersions)
            std::cout << "  " << version << "  x" << count << '\n';

        std::cout << "\n--- record types (instances / files containing) ---\n";
        for (const auto& [name, count] : gStats.mRecords)
        {
            const auto files = gStats.mRecordFiles.find(name);
            std::cout << "  " << name << "  " << count << " / "
                      << (files != gStats.mRecordFiles.end() ? files->second : 0u) << '\n';
        }

        std::cout << "\n(controller counts are in the record list above -- every controller is a record, and"
                     " counting them there is exact where walking the tree would count a shared property's"
                     " controller once per shape.)\n";

        std::cout << "\n--- texture slot combinations, per shape ---\n";
        for (const auto& [combo, bucket] : gStats.mSlotCombos)
            std::cout << "  " << combo << "  x" << bucket.mCount << '\n';

        std::cout << "\n--- apply modes ---\n";
        for (const auto& [name, count] : gStats.mApplyModes)
            std::cout << "  " << name << "  x" << count << '\n';

        std::cout << "\n--- shapes, and what the stage bake will do with them ---\n";
        std::cout << "  shapes total: " << gStats.mShapes << " (untextured " << gStats.mShapesUntextured
                  << ", no base slot " << gStats.mShapesNoBaseSlot << ")\n";
        std::cout << "  multi-stage (dark and/or detail): " << gStats.mShapesMultiStage << '\n';
        std::cout << "  carrying a UV controller: " << gStats.mShapesAnimatedUv << '\n';
        reportBucket("every baked stage on the base's uv set -- factorable, animation survives",
            gStats.mFactorable, nullptr);
        reportBucket("a baked stage on another uv set -- must fold, needs the per-triangle affine",
            gStats.mMustFold, nullptr);
        reportBucket("ANIMATED and must fold -- animation freezes until the phase sprite sheet exists",
            gStats.mAnimatedMustFold, "this is the count that sizes option B:");
        reportBucket("of those, ONE AFFINE describes the whole surface", gStats.mOneAffineFits,
            "expressible as a texture transform in a shader, so these would not need a baked image at all:");
        reportBucket("of those, genuinely PER-TRIANGLE", gStats.mPerTriangleRequired,
            "no single affine fits; only a per-triangle solve or a bake can combine these:");
        std::cout << "  worst residual among the fitting ones: " << gStats.mWorstFittingResidual
                  << " UV (the threshold is 1e-3; a fit and a failure differ by orders of magnitude here)\n";

        reportBucket("stage on another uv set whose CONTENT is identical to the base's",
            gStats.mIdenticalUvTrap,
            "coordinate equality says these share; the uv set index says they do not. Testing coordinates"
            " here is what made the logs-on-fire dark map scroll:");
        reportBucket("base stage not on the animated uv set", gStats.mBaseOffAnimatedSet,
            "the host reads its texture matrix from unit 0, so the animation lands on the wrong stage:");
        reportBucket("stage carrying its own texture transform", gStats.mStageTransform,
            "the bake samples stages through the per-triangle affine and does NOT apply a stage's own"
            " offset/scale/rotation, so these are mapped wrongly:");

        std::cout << "\n--- materials ---\n";
        std::cout << "  NiMaterialProperty total: " << gStats.mMaterials << ", with a non-black emissive: "
                  << gStats.mMaterialsEmissive << ", with alpha below one: "
                  << gStats.mMaterialsAlphaBelowOne << '\n';
        reportBucket("black diffuse with a non-black emissive -- purely emissive",
            gStats.mBlackDiffuseEmissive,
            "the rasteriser adds emissive to diffuse; a host that multiplies diffuse into the albedo renders"
            " these black:");

        std::cout << "\n--- alpha ---\n";
        std::cout << "  blended: " << gStats.mAlphaBlended << ", tested: " << gStats.mAlphaTested << '\n';
        for (const auto& [mode, count] : gStats.mBlendModes)
            std::cout << "  src/dst " << mode << "  x" << count << '\n';

        std::cout << "\n--- features that need their own decision ---\n";
        reportBucket("NiTextureEffect -- projected/environment texture, NOT implemented by the host",
            gStats.mTextureEffects, nullptr);
        reportBucket("particle systems", gStats.mParticleSystems, nullptr);
        reportBucket("skinned geometry", gStats.mSkinned, nullptr);
        reportBucket("NiSwitchNode -- day/night and similar variants", gStats.mSwitchNodes, nullptr);
        reportBucket("geometry with more than one uv set", gStats.mMultiUvGeometry, nullptr);

        std::cout << "\n--- textures ---\n";
        std::cout << "  distinct texture references: " << gStats.mTextures.size() << '\n';
        for (const auto& [ext, count] : gStats.mTextureExtensions)
            std::cout << "  " << ext << "  x" << count << '\n';
        std::cout << std::endl;
    }
}

std::unique_ptr<VFS::Archive> makeArchive(const std::filesystem::path& path)
{
    if (isBSA(path))
        return VFS::makeBsaArchive(path, nullptr);
    if (std::filesystem::is_directory(path))
        return std::make_unique<VFS::FileSystemArchive>(path);
    return nullptr;
}

/// Set by --dump/--filter/--census. File scope rather than threaded through every call site, because readFile
/// and readVFS are recursive and already carry a parameter each for the same reason.
bool gDump = false;
bool gCensus = false;
std::string gFilter;

bool readFile(
    const std::filesystem::path& source, const std::filesystem::path& path, const VFS::Manager* vfs, bool quiet)
{
    const auto [fileType, fileClass] = classifyFile(path);
    if (fileClass != FileClass::NIF && fileClass != FileClass::Material)
        return false;

    const std::string pathStr = Files::pathToUnicodeString(path);
    if (!quiet)
    {
        std::cout << "Reading " << getFileTypeName(fileType) << " file '" << pathStr << "'";
        if (!source.empty())
            std::cout << " from '" << Files::pathToUnicodeString(isBSA(source) ? source.filename() : source) << "'";
        std::cout << std::endl;
    }
    const std::filesystem::path fullPath = !source.empty() ? source / path : path;
    try
    {
        switch (fileClass)
        {
            case FileClass::NIF:
            {
                Nif::NIFFile file(VFS::Path::Normalized(Files::pathToUnicodeString(fullPath)));
                Nif::Reader reader(file, nullptr);
                if (vfs != nullptr)
                    reader.parse(vfs->get(VFS::Path::Normalized(pathStr)));
                else
                    reader.parse(Files::openConstrainedFileStream(fullPath));
                // After parse, so the records exist and post-processing has resolved the pointers a slot's
                // source texture is reached through.
                if (gDump)
                    NifDump::dump(file, gFilter);
                if (gCensus)
                    NifCensus::accumulate(file);
                break;
            }
            case FileClass::Material:
            {
                if (vfs != nullptr)
                    Bgsm::parse(vfs->get(VFS::Path::Normalized(pathStr)));
                else
                    Bgsm::parse(Files::openConstrainedFileStream(fullPath));
                break;
            }
            default:
                break;
        }
    }
    catch (std::exception& e)
    {
        // Counted as well as printed. A census over tens of thousands of files scrolls its failures off the
        // top of any terminal, and "how many did not parse" is the first thing that decides whether the
        // totals below it mean anything.
        if (gCensus)
            NifCensus::noteFailure();
        std::cerr << "Failed to read '" << pathStr << "':" << std::endl << e.what() << std::endl;
    }
    return true;
}

/// Check all the nif files in a given VFS::Archive
/// \note Can not read a bsa file inside of a bsa file.
void readVFS(std::unique_ptr<VFS::Archive>&& archive, const std::filesystem::path& archivePath, bool quiet)
{
    if (archive == nullptr)
        return;

    if (!quiet)
        std::cout << "Reading data source '" << Files::pathToUnicodeString(archivePath) << "'" << std::endl;

    VFS::Manager vfs;
    vfs.addArchive(std::move(archive));
    vfs.buildIndex();

    for (const auto& name : vfs.getRecursiveDirectoryIterator())
    {
        readFile(archivePath, name.value(), &vfs, quiet);
    }

    if (!archivePath.empty() && !isBSA(archivePath))
    {
        const Files::Collections fileCollections({ archivePath });
        const Files::MultiDirCollection& bsaCol = fileCollections.getCollection("bsa");
        const Files::MultiDirCollection& ba2Col = fileCollections.getCollection("ba2");
        for (const Files::MultiDirCollection& collection : { bsaCol, ba2Col })
        {
            for (auto& file : collection)
            {
                try
                {
                    readVFS(VFS::makeBsaArchive(file.second, nullptr), file.second, quiet);
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Failed to read archive file '" << Files::pathToUnicodeString(file.second)
                              << "': " << e.what() << std::endl;
                }
            }
        }
    }
}

bool parseOptions(int argc, char** argv, Files::PathContainer& files, Files::PathContainer& archives,
    bool& writeDebugLog, bool& quiet)
{
    bpo::options_description desc(
        R"(Ensure that OpenMW can use the provided NIF, KF, BTO/BTR, RDT, PSA, BGEM/BGSM and BSA/BA2 files

Usages:
  niftest <nif files, kf files, bto/btr files, rdt files, psa files, bgem/bgsm files, BSA/BA2 files, or directories>
      Scan the file or directories for NIF errors.

Allowed options)");
    auto addOption = desc.add_options();
    addOption("help,h", "print help message.");
    addOption("write-debug-log,v", "write debug log for unsupported nif files");
    addOption("quiet,q", "do not log read archives/files");
    addOption("dump,d",
        "print each NIF's structure: record histogram, NiTexturingProperty slots with their uvSet and "
        "transform, NiMaterialProperty colours, NiAlphaProperty flags. Read through OpenMW's own parser, "
        "so it reports what the engine loaded rather than a second interpretation of the format.");
    addOption("census,c",
        "survey every NIF read and print one aggregate report at the end instead of per-file output. "
        "Classifies each shape by what the Remix host's stage bake will do with it, counts the features the "
        "host does not implement, and names a few example files per bucket. Use this to size work across an "
        "installed asset set; use --dump to understand one mesh.");
    addOption("filter", bpo::value<std::string>(),
        "with --dump, restrict texturing properties to those referencing a texture whose name contains "
        "this substring. Case insensitive. Needed for merged meshes -- a town mesh holds hundreds of "
        "properties and only one of them is the question.");
    addOption("archives", bpo::value<Files::MaybeQuotedPathContainer>(), "path to archive files to provide files");
    addOption("input-file", bpo::value<Files::MaybeQuotedPathContainer>(), "input file");

    // Default option if none provided
    bpo::positional_options_description p;
    p.add("input-file", -1);

    bpo::variables_map variables;
    try
    {
        bpo::parsed_options validOpts = bpo::command_line_parser(argc, argv).options(desc).positional(p).run();
        bpo::store(validOpts, variables);
        bpo::notify(variables);
        if (variables.count("help"))
        {
            std::cout << desc << std::endl;
            return false;
        }
        writeDebugLog = variables.count("write-debug-log") > 0;
        quiet = variables.count("quiet") > 0;
        gDump = variables.count("dump") > 0;
        gCensus = variables.count("census") > 0;
        if (const auto it = variables.find("filter"); it != variables.end())
            gFilter = it->second.as<std::string>();
        if (variables.count("input-file"))
        {
            files = asPathContainer(variables["input-file"].as<Files::MaybeQuotedPathContainer>());
            if (const auto it = variables.find("archives"); it != variables.end())
                archives = asPathContainer(it->second.as<Files::MaybeQuotedPathContainer>());
            return true;
        }
    }
    catch (std::exception& e)
    {
        std::cout << "Error parsing arguments: " << e.what() << "\n\n" << desc << std::endl;
        return false;
    }

    std::cout << "No input files or directories specified!" << std::endl;
    std::cout << desc << std::endl;
    return false;
}

int main(int argc, char** argv)
{
    Files::PathContainer files, sources;
    bool writeDebugLog = false;
    bool quiet = false;
    if (!parseOptions(argc, argv, files, sources, writeDebugLog, quiet))
        return 1;

    Nif::Reader::setWriteNifDebugLog(writeDebugLog);

    std::unique_ptr<VFS::Manager> vfs;
    if (!sources.empty())
    {
        vfs = std::make_unique<VFS::Manager>();
        for (const std::filesystem::path& path : sources)
        {
            const std::string pathStr = Files::pathToUnicodeString(path);
            if (!quiet)
                std::cout << "Adding data source '" << pathStr << "'" << std::endl;

            try
            {
                if (auto archive = makeArchive(path))
                    vfs->addArchive(std::move(archive));
                else
                    std::cerr << "Error: '" << pathStr << "' is not an archive or directory" << std::endl;
            }
            catch (std::exception& e)
            {
                std::cerr << "Failed to add data source '" << pathStr << "':  " << e.what() << std::endl;
            }
        }

        vfs->buildIndex();
    }

    for (const auto& path : files)
    {
        const std::string pathStr = Files::pathToUnicodeString(path);
        try
        {
            const bool isFile = readFile({}, path, vfs.get(), quiet);
            if (!isFile)
            {
                if (auto archive = makeArchive(path))
                {
                    readVFS(std::move(archive), path, quiet);
                }
                else
                {
                    std::cerr << "Error: '" << pathStr << "' is not a NIF file, material file, archive, or directory"
                              << std::endl;
                }
            }
        }
        catch (std::exception& e)
        {
            std::cerr << "Failed to read '" << pathStr << "':  " << e.what() << std::endl;
        }
    }

    if (gCensus)
        NifCensus::report();

    return 0;
}
