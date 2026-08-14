#include "remixstagebake.hpp"

#include <algorithm>
#include <cmath>

namespace
{
    /// A 2D affine as three coefficients per output axis, evaluated as a*u + b*v + c.
    struct Affine
    {
        float mU[3] = { 1.0f, 0.0f, 0.0f };
        float mV[3] = { 0.0f, 1.0f, 0.0f };
    };

    /// The exact affine taking one triangle's base texcoords to the same triangle's stage texcoords.
    ///
    /// Three points determine an affine, so this is a solve rather than a fit: no least squares, no
    /// residual, nothing to accept or reject. That is the entire reason the work is done per triangle. A
    /// per-vertex UV mapping is piecewise linear, so it is affine inside every triangle even when no single
    /// affine describes the whole surface -- which is the case that defeated both earlier attempts at this.
    ///
    /// -> false when the triangle has no area in *base* UV space. Its stage UVs then cannot be placed,
    /// because the space being rasterised into has collapsed. That is a property of the asset, not of this
    /// arithmetic, and the caller counts them.
    bool solveTriangleAffine(const osg::Vec2f& b0, const osg::Vec2f& b1, const osg::Vec2f& b2,
        const osg::Vec2f& s0, const osg::Vec2f& s1, const osg::Vec2f& s2, Affine& out)
    {
        const double det = static_cast<double>(b0.x()) * (b1.y() - b2.y())
            - static_cast<double>(b0.y()) * (b1.x() - b2.x())
            + (static_cast<double>(b1.x()) * b2.y() - static_cast<double>(b2.x()) * b1.y());

        // Relative to the triangle's own size rather than absolute. Morrowind texcoords are order-one, but
        // a small pane's triangle can be a hundredth of that across, and an absolute epsilon would discard
        // legitimate geometry while still admitting a genuinely collapsed triangle on a large one.
        const double scale = std::max({ std::abs(static_cast<double>(b1.x() - b0.x())),
            std::abs(static_cast<double>(b1.y() - b0.y())), std::abs(static_cast<double>(b2.x() - b0.x())),
            std::abs(static_cast<double>(b2.y() - b0.y())), 1e-8 });
        if (std::abs(det) < 1e-9 * scale * scale)
            return false;

        const double invDet = 1.0 / det;
        // Cramer's rule on the 3x3 [u v 1] system, once per output axis.
        const auto solve = [&](float t0, float t1, float t2, float (&row)[3]) {
            const double a = static_cast<double>(t0) * (b1.y() - b2.y())
                - static_cast<double>(b0.y()) * (t1 - t2)
                + (static_cast<double>(t1) * b2.y() - static_cast<double>(t2) * b1.y());
            const double b = static_cast<double>(b0.x()) * (t1 - t2) - static_cast<double>(t0) * (b1.x() - b2.x())
                + (static_cast<double>(b1.x()) * t2 - static_cast<double>(b2.x()) * t1);
            const double c = static_cast<double>(b0.x()) * (static_cast<double>(b1.y()) * t2 - static_cast<double>(b2.y()) * t1)
                - static_cast<double>(b0.y()) * (static_cast<double>(b1.x()) * t2 - static_cast<double>(b2.x()) * t1)
                + static_cast<double>(t0) * (static_cast<double>(b1.x()) * b2.y() - static_cast<double>(b2.x()) * b1.y());
            row[0] = static_cast<float>(a * invDet);
            row[1] = static_cast<float>(b * invDet);
            row[2] = static_cast<float>(c * invDet);
        };
        solve(s0.x(), s1.x(), s2.x(), out.mU);
        solve(s0.y(), s1.y(), s2.y(), out.mV);
        return true;
    }

    /// Bilinear sample with wrapping, in normalised texture space.
    ///
    /// Wrapped rather than clamped, unlike the terrain blend-map sampler in remixscene.cpp: these stages
    /// tile by design -- a lattice repeats several times across a pane -- so clamping would smear the last
    /// row of texels across everything past the first repeat.
    osg::Vec4f sampleWrapped(const osg::Image& image, float s, float t)
    {
        const int width = image.s();
        const int height = image.t();
        if (image.data() == nullptr || width <= 0 || height <= 0)
            return osg::Vec4f(1.0f, 1.0f, 1.0f, 1.0f);

        const auto wrapIndex = [](int index, int extent) {
            const int m = index % extent;
            return m < 0 ? m + extent : m;
        };

        const float cs = s * static_cast<float>(width) - 0.5f;
        const float ct = t * static_cast<float>(height) - 0.5f;
        const int x0 = static_cast<int>(std::floor(cs));
        const int y0 = static_cast<int>(std::floor(ct));
        const float fx = cs - static_cast<float>(x0);
        const float fy = ct - static_cast<float>(y0);

        const auto at = [&image, &wrapIndex, width, height](int x, int y) {
            return image.getColor(static_cast<unsigned int>(wrapIndex(x, width)),
                static_cast<unsigned int>(wrapIndex(y, height)));
        };

        const osg::Vec4f top = at(x0, y0) * (1.0f - fx) + at(x0 + 1, y0) * fx;
        const osg::Vec4f bottom = at(x0, y0 + 1) * (1.0f - fx) + at(x0 + 1, y0 + 1) * fx;
        return top * (1.0f - fy) + bottom * fy;
    }
}

namespace MWRender::RemixStageBake
{
    Result combine(const osg::Image& base, const osg::Vec2Array& baseCoords,
        const std::vector<unsigned int>& triangles, const std::vector<Stage>& stages,
        const std::vector<std::array<float, 6>>& basePhases, unsigned int maxDimension,
        unsigned int maxSheetWidth)
    {
        Result result;

        // One phase is the ordinary case and an empty list means the same thing, so neither is an error.
        static const std::vector<std::array<float, 6>> identityPhase{ { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f } };
        const std::vector<std::array<float, 6>>& phases = basePhases.empty() ? identityPhase : basePhases;

        if (stages.empty())
        {
            result.mReason = "no stages beyond the base";
            return result;
        }
        if (base.data() == nullptr || base.s() <= 0 || base.t() <= 0)
        {
            result.mReason = "the base image has no decoded texels";
            return result;
        }
        if (triangles.size() < 3)
        {
            result.mReason = "no triangles, so no per-triangle mapping is possible";
            return result;
        }
        for (const Stage& stage : stages)
        {
            if (stage.mImage == nullptr || stage.mCoords == nullptr)
            {
                result.mReason = std::string("stage '") + stage.mName + "' has no image or no texcoords";
                return result;
            }
            if (stage.mCoords->size() != baseCoords.size())
            {
                result.mReason = std::string("stage '") + stage.mName + "' has "
                    + std::to_string(stage.mCoords->size()) + " texcoords against the base's "
                    + std::to_string(baseCoords.size());
                return result;
            }
        }

        // Every stage moves with the base, so the base's matrix is a common factor of the product and comes
        // straight back out of it: base(Mt) * stage(Mt) is f(Mt) for a single f, and f is what is baked. The
        // matrix is left for the caller to keep on the surface, so an animated one goes on animating.
        //
        // None of the triangle machinery below applies here. Sharing the base's uv set means the mapping from
        // base coords to stage coords is the identity at every vertex and so everywhere between them, which
        // makes the product a per-texel function of UV across the whole square: no affine to solve, no
        // coverage to rasterise, no seams between triangles, and nothing that can come out degenerate.
        if (std::all_of(
                stages.begin(), stages.end(), [](const Stage& stage) { return stage.mSharesBaseTransform; }))
        {
            // The full UV square rather than the drawable's footprint. The caller's matrix is still live and
            // can take the sampled coordinate anywhere, so the baked image has to be defined everywhere the
            // sources are, and it has to wrap with the sources' period: a pane whose footprint spans 0.986 of
            // the atlas would otherwise repeat every 0.986 and visibly jump once per scroll cycle.
            //
            // Resolution needs no estimating, for the same reason the affines need no solving. An identity
            // mapping magnifies nothing, so one texel out is one texel of the sharpest input.
            int width = base.s();
            int height = base.t();
            for (const Stage& stage : stages)
            {
                width = std::max(width, stage.mImage->s());
                height = std::max(height, stage.mImage->t());
            }
            width = std::clamp(width, 4, static_cast<int>(maxDimension));
            height = std::clamp(height, 4, static_cast<int>(maxDimension));

            osg::ref_ptr<osg::Image> factored = new osg::Image;
            factored->allocateImage(width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE);
            if (factored->data() == nullptr)
            {
                result.mReason = "could not allocate the combined image";
                return result;
            }
            factored->setInternalTextureFormat(GL_RGBA8);

            for (int y = 0; y < height; ++y)
            {
                unsigned char* row = factored->data(0, y);
                for (int x = 0; x < width; ++x)
                {
                    const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
                    const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

                    osg::Vec4f colour = sampleWrapped(base, u, v);
                    for (const Stage& stage : stages)
                    {
                        const osg::Vec4f sampled = sampleWrapped(*stage.mImage, u, v);
                        // Colour only, and alpha left to the base, as in the folded path below.
                        for (int channel = 0; channel < 3; ++channel)
                            colour[channel] *= sampled[channel] * stage.mMultiplier;
                    }

                    for (int channel = 0; channel < 4; ++channel)
                        row[4 * x + channel]
                            = static_cast<unsigned char>(std::clamp(colour[channel], 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }

            result.mImage = factored;
            result.mBaseTransformFactoredOut = true;
            // One frame, whatever was asked for. This path does not sample the transform at all -- it factors
            // out and stays live on the surface -- so phases of it would be identical images.
            result.mSpriteFrames = 1;
            return result;
        }

        // The footprint actually used, so the bake describes the pane rather than the atlas it sits in.
        float minU = baseCoords[0].x();
        float maxU = minU;
        float minV = baseCoords[0].y();
        float maxV = minV;
        for (const osg::Vec2f& coord : baseCoords)
        {
            minU = std::min(minU, coord.x());
            maxU = std::max(maxU, coord.x());
            minV = std::min(minV, coord.y());
            maxV = std::max(maxV, coord.y());
        }
        constexpr float kMinExtent = 1e-6f;
        const float extentU = std::max(maxU - minU, kMinExtent);
        const float extentV = std::max(maxV - minV, kMinExtent);

        // Solve every triangle up front, so resolution can be chosen from the mappings and the raster pass
        // has nothing left to decide.
        const std::size_t triangleCount = triangles.size() / 3;
        std::vector<std::vector<Affine>> perTriangle(triangleCount);
        std::vector<bool> usable(triangleCount, false);

        // Resolution from whichever stage varies fastest anywhere on the surface, base included. The base is
        // frequently the coarsest of the set -- a gradient under a lattice -- so sizing to it is what would
        // lose the detail this exists to recover.
        double neededWidth = static_cast<double>(base.s()) * extentU;
        double neededHeight = static_cast<double>(base.t()) * extentV;

        for (std::size_t t = 0; t < triangleCount; ++t)
        {
            const unsigned int i0 = triangles[t * 3 + 0];
            const unsigned int i1 = triangles[t * 3 + 1];
            const unsigned int i2 = triangles[t * 3 + 2];
            if (i0 >= baseCoords.size() || i1 >= baseCoords.size() || i2 >= baseCoords.size())
            {
                ++result.mTrianglesDegenerate;
                continue;
            }

            std::vector<Affine> affines(stages.size());
            bool ok = true;
            for (std::size_t s = 0; s < stages.size() && ok; ++s)
            {
                const osg::Vec2Array& sc = *stages[s].mCoords;
                ok = solveTriangleAffine(baseCoords[i0], baseCoords[i1], baseCoords[i2], sc[i0], sc[i1],
                    sc[i2], affines[s]);
            }
            if (!ok)
            {
                ++result.mTrianglesDegenerate;
                continue;
            }

            for (std::size_t s = 0; s < stages.size(); ++s)
            {
                const double spanU = std::abs(affines[s].mU[0]) * extentU + std::abs(affines[s].mU[1]) * extentV;
                const double spanV = std::abs(affines[s].mV[0]) * extentU + std::abs(affines[s].mV[1]) * extentV;
                neededWidth = std::max(neededWidth, static_cast<double>(stages[s].mImage->s()) * spanU);
                neededHeight = std::max(neededHeight, static_cast<double>(stages[s].mImage->t()) * spanV);
            }

            perTriangle[t] = std::move(affines);
            usable[t] = true;
            ++result.mTrianglesMapped;
        }

        if (result.mTrianglesMapped == 0)
        {
            result.mReason = "every triangle has zero area in base UV space, so there is nowhere to bake into";
            return result;
        }

        // How many phases fit side by side, and how wide each may be.
        //
        // The sheet is 1 x N because that is what the runtime wrapper declares -- createTexturedMaterial
        // hardcodes a single row -- so the phase count multiplies the width and nothing else. Raising it
        // therefore lowers the frame resolution rather than growing the sheet past what a driver will accept,
        // and that is the entire tradeoff: more phases means smoother motion and a coarser stage pattern.
        //
        // Eight phases at the default 1,024 cap comes to exactly the 8,192 budget, which is why the default
        // costs no resolution at all compared with a single frozen frame.
        int frames = static_cast<int>(std::min<std::size_t>(std::max<std::size_t>(phases.size(), 1), 255));
        int widthCap = static_cast<int>(maxDimension);
        if (frames > 1)
            widthCap = std::min(widthCap, std::max(4, static_cast<int>(maxSheetWidth) / frames));

        const auto clampTo = [](double needed, int cap) {
            return static_cast<int>(std::clamp(std::ceil(needed), 4.0, static_cast<double>(cap)));
        };
        const int width = clampTo(neededWidth, widthCap);
        const int height = clampTo(neededHeight, static_cast<int>(maxDimension));

        // Recomputed now that the width is known. A frame narrower than its cap leaves room for more phases
        // than the cap assumed, and a frame that hit the floor of four could still overflow a small budget.
        frames = std::max(1, std::min(frames, static_cast<int>(maxSheetWidth) / std::max(1, width)));
        const int sheetWidth = width * frames;

        osg::ref_ptr<osg::Image> out = new osg::Image;
        out->allocateImage(sheetWidth, height, 1, GL_RGBA, GL_UNSIGNED_BYTE);
        if (out->data() == nullptr)
        {
            result.mReason = "could not allocate the combined image";
            return result;
        }
        out->setInternalTextureFormat(GL_RGBA8);
        result.mSpriteFrames = static_cast<unsigned int>(frames);

        const auto uvForTexel = [&](int x, int y) {
            // Texel centres, so the sampled range matches what a GPU would fetch for this footprint.
            return osg::Vec2f(minU + (static_cast<float>(x) + 0.5f) / static_cast<float>(width) * extentU,
                minV + (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * extentV);
        };

        // Pixel space is per frame, not per sheet: the triangles are rasterised into a frame's own
        // coordinates and only the write is offset into that frame's columns. Hoisted above the phase loop
        // because neither depends on the phase.
        const float toPixelU = static_cast<float>(width) / extentU;
        const float toPixelV = static_cast<float>(height) / extentV;

        // One frame per phase of the animation, laid out left to right in the order the runtime plays them.
        for (int frame = 0; frame < frames; ++frame)
        {
            // Spread across the phases supplied rather than taking the first N of them.
            //
            // Only matters when fewer frames fit than were asked for, which the width arithmetic above
            // currently prevents -- but truncating would be wrong in a way that is hard to see: the frames
            // would cover the start of the animation's loop while the runtime played them as the whole loop,
            // so the motion would run over part of its range and at the wrong speed. The caller derives the
            // playback rate from the full period, so the frames kept have to span the full period too.
            const std::size_t phaseIndex
                = static_cast<std::size_t>(frame) * phases.size() / static_cast<std::size_t>(frames);
            const std::array<float, 6>& baseTransform = phases[std::min(phaseIndex, phases.size() - 1)];
            const int x0 = frame * width;

            // Pass one: the base everywhere.
            //
            // Every texel gets a defined value before any triangle is considered, which matters for the ones
            // no triangle covers -- the footprint is a bounding box and the geometry inside it need not fill
            // it. Those texels then hold base-only colour, which is what this surface looked like before any
            // of this existed. Leaving them uninitialised would put garbage in the gaps, and clearing them to
            // black would draw seams the asset does not have.
            for (int y = 0; y < height; ++y)
            {
                unsigned char* row = out->data(0, y);
                for (int x = 0; x < width; ++x)
                {
                    const osg::Vec2f uv = uvForTexel(x, y);
                    const float bu = baseTransform[0] * uv.x() + baseTransform[2] * uv.y() + baseTransform[4];
                    const float bv = baseTransform[1] * uv.x() + baseTransform[3] * uv.y() + baseTransform[5];
                    const osg::Vec4f colour = sampleWrapped(base, bu, bv);
                    for (int channel = 0; channel < 4; ++channel)
                        row[4 * (x0 + x) + channel] = static_cast<unsigned char>(
                            std::clamp(colour[channel], 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }

            // Pass two: multiply each stage in, triangle by triangle, through that triangle's own affine.
            for (std::size_t t = 0; t < triangleCount; ++t)
            {
                if (!usable[t])
                    continue;

                const unsigned int i0 = triangles[t * 3 + 0];
                const unsigned int i1 = triangles[t * 3 + 1];
                const unsigned int i2 = triangles[t * 3 + 2];

                // The triangle in the baked image's pixel space.
                const osg::Vec2f p0((baseCoords[i0].x() - minU) * toPixelU, (baseCoords[i0].y() - minV) * toPixelV);
                const osg::Vec2f p1((baseCoords[i1].x() - minU) * toPixelU, (baseCoords[i1].y() - minV) * toPixelV);
                const osg::Vec2f p2((baseCoords[i2].x() - minU) * toPixelU, (baseCoords[i2].y() - minV) * toPixelV);

                const int xLow = std::max(0, static_cast<int>(std::floor(std::min({ p0.x(), p1.x(), p2.x() }))) - 1);
                const int xHigh = std::min(width - 1, static_cast<int>(std::ceil(std::max({ p0.x(), p1.x(), p2.x() }))) + 1);
                const int yLow = std::max(0, static_cast<int>(std::floor(std::min({ p0.y(), p1.y(), p2.y() }))) - 1);
                const int yHigh = std::min(height - 1, static_cast<int>(std::ceil(std::max({ p0.y(), p1.y(), p2.y() }))) + 1);

                const float area = (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p2.x() - p0.x()) * (p1.y() - p0.y());
                if (std::abs(area) < 1e-12f)
                    continue;
                const float invArea = 1.0f / area;

                // How far outside each edge a texel centre may sit and still be filled, expressed as the
                // barycentric weight that a slack of kSlack *pixels* corresponds to.
                //
                // The conversion is not optional. A barycentric weight is an area ratio, so the same weight
                // means a different distance on every triangle: a weight of w sits at a perpendicular distance
                // of w * area / edgeLength from that edge. Comparing weights against a fixed number instead
                // would scale the slack by the triangle's shape -- on a pane-sized triangle a flat -0.75/area
                // limit works out to some three thousandths of a texel, which is the strict test this exists to
                // avoid, while on a sliver it would reach far outside the triangle.
                constexpr float kSlack = 0.75f;
                const float absArea = std::max(std::abs(area), 1e-12f);
                const auto slackLimit = [&](const osg::Vec2f& a, const osg::Vec2f& b) {
                    const float length = std::sqrt((b.x() - a.x()) * (b.x() - a.x()) + (b.y() - a.y()) * (b.y() - a.y()));
                    return -kSlack * length / absArea - 1e-6f;
                };
                // Each weight is bounded by the edge opposite its vertex.
                const float limit0 = slackLimit(p1, p2);
                const float limit1 = slackLimit(p2, p0);
                const float limit2 = slackLimit(p0, p1);

                for (int y = yLow; y <= yHigh; ++y)
                {
                    unsigned char* row = out->data(0, y);
                    for (int x = xLow; x <= xHigh; ++x)
                    {
                        const float px = static_cast<float>(x) + 0.5f;
                        const float py = static_cast<float>(y) + 0.5f;

                        // Barycentric coverage, with the per-edge slack computed above.
                        //
                        // The slack closes the gaps a strict test leaves along shared edges, and it costs
                        // nothing: two triangles sharing an edge agree at both its endpoints, and two affines
                        // agreeing at two points agree everywhere on the line through them. So the overlap
                        // writes the same value twice rather than fighting over it. Without the slack a texel
                        // whose centre falls just outside every triangle keeps base-only colour and shows as a
                        // thin seam of the wrong colour.
                        const float w0 = ((p1.x() - px) * (p2.y() - py) - (p2.x() - px) * (p1.y() - py)) * invArea;
                        const float w1 = ((p2.x() - px) * (p0.y() - py) - (p0.x() - px) * (p2.y() - py)) * invArea;
                        const float w2 = 1.0f - w0 - w1;
                        if (w0 < limit0 || w1 < limit1 || w2 < limit2)
                            continue;

                        const osg::Vec2f uv = uvForTexel(x, y);

                        // This frame's column. Pass one already wrote the base here, so the read below picks it
                        // up and the stages multiply into it.
                        const int sx = x0 + x;

                        osg::Vec4f colour(row[4 * sx + 0] / 255.0f, row[4 * sx + 1] / 255.0f,
                            row[4 * sx + 2] / 255.0f, row[4 * sx + 3] / 255.0f);

                        // Where the base is being read from, needed again below for any stage that moves with it.
                        const float bu
                            = baseTransform[0] * uv.x() + baseTransform[2] * uv.y() + baseTransform[4];
                        const float bv
                            = baseTransform[1] * uv.x() + baseTransform[3] * uv.y() + baseTransform[5];

                        for (std::size_t s = 0; s < stages.size(); ++s)
                        {
                            const Affine& affine = perTriangle[t][s];
                            // A stage on the base's uv set is read at the base's transformed coordinate, not at
                            // the untransformed one. Its affine is the identity -- that is what sharing the uv
                            // set means -- so the rasteriser reads it at Mt exactly as it reads the base.
                            //
                            // Getting this wrong is not subtle. A soul gem's glow puts base and detail on one
                            // texture and one uvSet, so sampling the stage untransformed multiplies a still copy
                            // of the image against a moving one: a ghost, rather than the self-multiply the asset
                            // asks for. Those surfaces take the factored path above, but a mixed state set with
                            // one shared stage and one on another uv set has to fold, and still needs this.
                            const float su = stages[s].mSharesBaseTransform
                                ? bu
                                : affine.mU[0] * uv.x() + affine.mU[1] * uv.y() + affine.mU[2];
                            const float sv = stages[s].mSharesBaseTransform
                                ? bv
                                : affine.mV[0] * uv.x() + affine.mV[1] * uv.y() + affine.mV[2];
                            const osg::Vec4f sampled = sampleWrapped(*stages[s].mImage, su, sv);

                            // Multiplied per channel and scaled, which is what the rasteriser does: a dark map
                            // at one and a detail map at two. Alpha is left to the base -- a stage's alpha is
                            // not part of that combination, and taking it would make an opaque surface
                            // transparent wherever a detail map happened to carry one.
                            for (int channel = 0; channel < 3; ++channel)
                                colour[channel] *= sampled[channel] * stages[s].mMultiplier;
                        }

                        // Clamped rather than allowed to wrap: a detail map at two is deliberately able to
                        // exceed one, and an eight-bit target has to stop somewhere, as the rasteriser's own
                        // framebuffer does.
                        for (int channel = 0; channel < 3; ++channel)
                            row[4 * sx + channel]
                                = static_cast<unsigned char>(std::clamp(colour[channel], 0.0f, 1.0f) * 255.0f + 0.5f);
                    }
                }
            }
        }

        // Base texcoord -> baked image, row-vector convention, matching SurfaceState::mTexMat.
        result.mTexMat = { 1.0f / extentU, 0.0f, 0.0f, 1.0f / extentV, -minU / extentU, -minV / extentV };
        result.mImage = out;
        return result;
    }
}
