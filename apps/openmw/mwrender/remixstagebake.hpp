#ifndef OPENMW_MWRENDER_REMIXSTAGEBAKE_H
#define OPENMW_MWRENDER_REMIXSTAGEBAKE_H

#include <array>
#include <string>
#include <vector>

#include <osg/Array>
#include <osg/Image>
#include <osg/ref_ptr>

namespace MWRender::RemixStageBake
{
    /// Combines a drawable's texture stages into one albedo image.
    ///
    /// A path-traced surface has one albedo. A Morrowind NiTexturingProperty can have up to seven texture
    /// stages, and OpenMW's rasteriser combines them per fragment -- base, then `*= darkMap`, then
    /// `*= detailMap * 2.0` (objects.frag L135, L156, L176). Remix has nowhere to put the second and third,
    /// so the host consumed the base and dropped the rest.
    ///
    /// On the Glow in the Dahrk night windows that lost nearly everything, and the measurements say so
    /// plainly. The base every one of them samples is glow/tex02_dark1.dds, a greyscale gradient averaging
    /// (111,111,111). The colour is on the *dark* stage -- tex02_2.dds averages (51,25,5), the amber, and
    /// tex03_2.dds averages (11,95,75), the green of the glass variants. The window pattern is on the
    /// *detail* stage. So a pane rendered from the base alone is a grey gradient with no colour and no
    /// pattern, which glowing reads as a flat white panel, and that is exactly what those windows looked
    /// like. Both the colour and the texture arrive only when the stages are combined.
    ///
    /// Combination is done per triangle, and that choice is the whole robustness of this.
    ///
    /// Each stage samples its own uvSet. Those sets are related to the base's by the mesh, and *globally*
    /// that relationship may be anything: on ex_common_window_01 the lattice's set is the base's scaled by
    /// (-26.67, -4.46), which is a transform, while on ex_redoran_window_01 the sets are a different
    /// projection entirely and no single transform describes them. An earlier attempt on this project
    /// flattened stages assuming the former and was reverted -- the visible failure was a lattice appearing
    /// flipped and magnified, which is what ignoring a scale of -26 looks like. A later attempt solved one
    /// affine per stage and verified it, which fixed the transform case and had to refuse the projection
    /// case, leaving those windows blank.
    ///
    /// Per triangle there is no such distinction. A per-vertex UV mapping is piecewise linear by
    /// construction, so within one triangle the map from base UV to stage UV is exactly affine and is
    /// determined by its three corners -- no fitting, no tolerance, no residual, and nothing to refuse.
    /// Adjacent triangles agree along their shared edge for free, because two affines that agree at two
    /// points agree everywhere on the line through them, so the seams cost nothing either.
    ///
    /// Animation is followed when it can be, and only frozen when it cannot. See mSharesBaseTransform.
    struct Stage
    {
        /// Decoded texels. Any format osg::Image::getColor understands.
        const osg::Image* mImage = nullptr;
        /// This stage's own texcoords, one per vertex, indexed by the same triangle list as the base's.
        const osg::Vec2Array* mCoords = nullptr;
        /// Applied after sampling. The rasteriser multiplies a detail map by two; a dark map by one.
        float mMultiplier = 1.0f;
        /// For the log.
        const char* mName = "";

        /// Whether this stage is moved by the same texture matrix as the base, because it reads the base's
        /// uv set. Set by the caller; it is a property of the state set, not of these texels.
        ///
        /// This is the difference between a soul gem that swirls and one frozen mid-swirl. Its glow shape
        /// puts base and detail on the *same* texture and the *same* uvSet, animated by a NiUVController, so
        /// the rasteriser computes glow(Mt) * glow(Mt) -- one matrix moving both. Fold that matrix in and
        /// only the base copy moves, leaving a still ghost of the same image multiplied against a moving
        /// one, which is neither the animation nor a clean freeze.
        ///
        /// When every stage sets this, the matrix is a common factor of the entire product: the result is
        /// f(Mt) for a single f, so f is baked once and the matrix is left on the surface, and an animated
        /// one goes on animating at no per-frame cost. When some stage does not -- the Glow in the Dahrk
        /// windows, whose base, dark and detail stages sit on three different uv sets -- there is no such
        /// factorisation and the matrix is folded in, freezing it. See baseTransform.
        bool mSharesBaseTransform = false;
    };

    struct Result
    {
        /// Null when nothing could be produced; mReason says why. Note that a stage being awkwardly mapped
        /// is no longer among the reasons.
        osg::ref_ptr<osg::Image> mImage;

        /// Maps a base texcoord into the baked image, as { m00, m01, m10, m11, m30, m31 } applied to a row
        /// vector -- the layout SurfaceState::mTexMat uses.
        ///
        /// Needed because the bake covers only the texcoord range the drawable actually uses. A window pane
        /// occupies about a fifth of its atlas in each axis while its lattice tiles three times across that,
        /// so baking the atlas's full [0,1] at a resolution that still resolves the lattice would cost tens
        /// of thousands of texels per axis to describe a region needing hundreds.
        ///
        /// The identity when the transform was factored out, that path covering the full UV square already.
        std::array<float, 6> mTexMat = { 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f };

        /// Triangles that contributed, and those skipped for zero area in base UV space. A high skip count
        /// means the drawable's base texcoords are degenerate -- collapsed to a point or a line -- which is
        /// worth seeing rather than silently producing a partly-base-only image.
        ///
        /// Both are zero when the transform was factored out, because that path needs no triangles: with
        /// every stage on the base's uv set the mapping between them is the identity everywhere, so there is
        /// no per-triangle solve to do and nothing that can be degenerate.
        unsigned int mTrianglesMapped = 0;
        unsigned int mTrianglesDegenerate = 0;

        /// Playback rate for the sheet, frames per second, 0 when this is not a sheet.
        ///
        /// Filled by the caller rather than by combine, which knows nothing about time -- it is derived from
        /// the animation's own period so the sheet runs at the speed the asset asks for. Cached alongside the
        /// image because both are stable for a given bake and recomputing it per drawable would mean carrying
        /// the controller into every lookup.
        unsigned char mSpriteFps = 0;

        /// How many animation phases the image holds, laid out left to right as a 1 x N sprite sheet.
        ///
        /// One means an ordinary single image. Above one, the caller must declare it to the runtime as a sprite
        /// sheet of this many columns, and must **not** rescale the surface's texcoords to suit: the runtime
        /// does that itself, `textureCoordinates = uvBias + frac(textureCoordinates) * uvSize` in
        /// surface_interaction.slangh, so mTexMat stays the plain footprint remap into one frame's [0,1].
        ///
        /// May be lower than the number of phases requested. The sheet grows sideways, so the phase count is
        /// bounded by the width budget, and the count that actually fit is reported here rather than assumed.
        unsigned int mSpriteFrames = 0;

        /// True when baseTransform was *not* folded into the image and must stay on the surface.
        ///
        /// The caller has to honour this, and the two cases are not interchangeable. Folded in, the returned
        /// mTexMat is the whole transform the finished image needs and replaces whatever the surface had.
        /// Factored out, the surface's own matrix is still live and mTexMat composes on top of it -- replace
        /// it there and the animation is discarded, which is the frozen soul gem this exists to avoid.
        bool mBaseTransformFactoredOut = false;

        /// Empty on success.
        std::string mReason;
    };

    /// Combines the stages into one image.
    ///
    /// \a triangles is a flat list of vertex indices, three per triangle, indexing all the coord arrays.
    /// osg::TriangleIndexFunctor produces exactly this and already decomposes strips, fans and quads.
    ///
    /// \a baseTransform is the texture matrix the base stage is sampled through, in the same row-vector
    /// layout. What happens to it depends on whether every stage shares it, and Result::
    /// mBaseTransformFactoredOut reports which of the two was done.
    ///
    /// Shared by every stage, it is a common factor: base(Mt) * detail(Mt) is f(Mt) for a single f, so f is
    /// baked in *untransformed* UV space and the matrix stays on the surface. An animated matrix then keeps
    /// animating, at no per-frame cost, because the image it moves over is phase-independent. This path
    /// covers the whole [0,1] UV square rather than the drawable's footprint, so that the surface can be
    /// sampled anywhere the matrix takes it and wrapping the baked image behaves as wrapping the sources
    /// would -- a footprint of 0.986 would wrap with the wrong period and jump once per scroll cycle.
    ///
    /// Not shared, and it has to be folded in: applied to the base's samples, and to those stages that do
    /// share it, while the rest are sampled untransformed. A single baked image cannot animate part of
    /// itself, so leaving the matrix on the surface instead would move the parts that were never animated.
    /// That was the first attempt on the Glow in the Dahrk windows, whose base carries a NiUVController and
    /// whose lattice does not, and the lattice scrolled and reversed along with the gradient.
    ///
    /// The consequence of folding, stated plainly: an animated base is frozen at whatever phase it held when
    /// the bake ran, and callers must not key a cache on this transform, since an animating value would bake
    /// a new image every frame. Freezing is the right answer there only because that base is a near-uniform
    /// grey gradient whose motion is not visible; it is the wrong answer for a structured glow, which is why
    /// the factored path exists.
    ///
    /// \a maxDimension caps either axis. Deliberately well below what full resolution would ask for: a
    /// window pane's lattice tiles about five times across its footprint, so resolving every texel of a
    /// 512-wide lattice would want 2,560 across -- eight megabytes for one pane, times every distinct window
    /// type. At 1,024 each repeat still gets some two hundred texels, far more than a pane a hundred pixels
    /// tall on screen can show, for two megabytes. Raise it if a pattern ever looks soft; the bake logs its
    /// size.
    /// \a maxSheetWidth caps the finished image's width, which is what bounds the phase count: the sheet is
    /// laid out 1 x N, so N frames of W each need N*W. 8,192 is the same budget spriteSheetFor uses for
    /// texture flipbooks, chosen because a texture the driver refuses to create is a worse outcome than a
    /// coarser one. Frames are narrowed to fit rather than the budget being exceeded.
    Result combine(const osg::Image& base, const osg::Vec2Array& baseCoords,
        const std::vector<unsigned int>& triangles, const std::vector<Stage>& stages,
        const std::vector<std::array<float, 6>>& basePhases, unsigned int maxDimension = 1024u,
        unsigned int maxSheetWidth = 8192u);
}

#endif
