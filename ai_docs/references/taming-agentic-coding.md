Taming agentic engineering - Prompts are code, .json/.md files are state
2025-06-02


Hello, Computer, do my work
Like most of you, I've been dabbling in what people call "agentic engineering." Truth is, there's not much engineering happening. We're basically throwing shit at the wall and hoping something sticks.

Using LLM coding tools like Claude Code to spin up throwaway greenfield projects or bang out ad hoc scripts? Pretty great experience. But try using them on a big, established codebase, or the-production-app-formerly-known-as-greenfield-project without breaking everything? That's where things get painful.

Feeling the pain
For bigger codebases, the main issue is context, or rather, the lack of it. These tools don't have the full picture of your project. Maybe you haven't given them that overview, or maybe their context window is too small to hold all your interconnected components. But there's more to it.

Even with recent improvements like 'reasoning', which is really just the old 'think step by step' trick, with more scratch space to attend to, LLMs still can't follow execution flow all that well. They're especially lost with anything beyond sequential scripts: multiple processes, IPC, client-server architectures, concurrent execution within the same process. Even when you manage to cram all the context they need, they'll still generate code that doesn't fit your system's actual architecture.

LLMs also lack taste. Trained on all code on the web (and likely some private code), they generate, to oversimplify, the statistical mean of what they've seen. While senior engineers strive for elegant, minimal solutions that reduce bugs and complexity, LLMs reach for 'best practices' and spit out over-engineered garbage. Let them run wild and you'll get code that's hard to maintain, hard to understand, and full of places for bugs to hide.

Then there's context degradation. As your session progresses and pulls in more files, tool outputs, and other data, things start falling apart around 100k tokens. Benchmarks be damned. Whatever tricks LLM providers use to achieve those massive context windows don't work in practice. The model loses track of important details buried in the middle of all that context.

Worse still, many tools don't let you control what goes into your context. Companies like Cursor that aren't LLM providers themselves need to make a margin between what you pay them and what they pay for tokens. Their incentive? Cut down your context to save money, which means the LLM might miss crucial information or get it in a suboptimal format.

Claude Code is different. It comes straight from Anthropic with no middleman trying to squeeze margins. With the Max plan, you get essentially unlimited tokens (though folks like Peter manage to get rate limited even with three or four accounts). You still don't have full control: there's a system prompt you can't change, additional instructions get sneakily injected into your first message, the VS Code integration adds unwanted crap, and all the tool definitions eat up context and give the model plenty of rope to confuse itself with. But this is the best deal we're getting, so we work with what we have. (Anthropic, please OSS Claude Code. Your models are your moat, not Claude Code.)

How do we tame this agentic mess?
What we need when using coding agents on bigger codebases is a structured way to engineer context. By that I mean: keep only the information needed for the task of modifying or generating code, minimize the number of turns the model needs to take calling tools or reporting back to us, and ensure nothing important is missing. We want reproducible workflows. We want determinism, as much as possible within the limits of these inherently non-deterministic models.

I'm a programmer. You're probably a programmer. We think in systems, deterministic workflows, and abstractions. What's more natural for us than viewing LLMs as an extremely slow kind of unreliable computer that we program with natural language?

This is a weird form of metaprogramming: we write "code" in the form of prompts that execute on the LLM to produce the actual code that runs on real CPUs.

Yes, I know LLMs aren't actually computers (though there are some papers on arXiv...). The metaphor is a bit stretched. But here's the thing: as developers, we're used to encoding specifications in precise programming languages. When we interact with LLMs, the fuzziness of natural language makes us forget we can apply the same structured thinking. This framework bridges that gap: think "inputs, state, outputs" instead of "chat with the AI" and suddenly you're closer to engineering solutions instead of just hoping for the best.

Thinking of LLMs as Shitty General Purpose Computers
In traditional software, we create programs by writing code and importing libraries. A program takes inputs, manipulates state, and produces outputs. We can map these concepts to our LLM-as-shitty-computer metaphor like this:

Program is your prompt, written in natural language. It specifies initial inputs, "imports" external functions via tool descriptions, and implements business logic through control flow: sequential steps, loops, conditionals, and yes, even goto. Tool calls and user input are I/O.

Inputs come from three sources: prepared information (codebase docs, style guides, architecture overviews) either baked into the prompt or loaded from disk, user input during execution (clarifications, corrections, new requirements), and tool outputs (file contents, command results, API responses).

State evolves as the program runs. Some lives in the context, but we treat that as ephemeral: compaction will eventually wipe it (trololo). Plus, you'll quickly hit context limits with any substantial state. So we serialize to disk using formats LLMs handle well: JSON for structured data, where the LLM can surgically read and update specific fields via jq. Markdown for smaller unstructured data we can load fully into context if needed. The payoff? You can resume from any point with a fresh context, sidestepping the dreaded compaction issue entirely.

Outputs aren't limited to generated code. Just like traditional programs produce console output, write files, or display GUIs, our LLM program uses tool calls to create various outputs: the actual code, diffs, open files in an editor for us, codebase statistics, summaries of changes, or any other artifact that documents what the program did. These outputs serve multiple purposes: helping you review the work, providing input for the next steps in the workflow, or simply showing the program's progress.

Let's see how this plays out in practice.

A Real World Example: Porting the Spine Runtimes
After experimenting with toy projects, I felt ready to apply this approach to a real codebase: the Spine runtimes.

Spine is 2D skeletal animation software. You create animations in the editor, export them to a runtime format, then use one of many runtimes to display them in your app or game. We maintain runtimes for C, C++, C#, Haxe, Java, Dart, Swift, and TypeScript. On top of these, we've built integrations for Unity, Unreal, Godot, Phaser, Pixi, ThreeJS, iOS, Android, web, and more.

Here's the painful part: between releases, the runtime code changes. We implement new features in our reference implementation (spine-libgdx in Java, which powers the editor), then manually port those changes to every other language runtime. It's tedious, error-prone work. Math-heavy code needs exact translation, and after hours of porting, your brain turns to mush. Bugs creep in that are hell to track down.

Git diff showing thousands of lines of code changes
And no, transpilers won't work for this (trust me, bro, I made money doing compilers). We need idiomatic ports that preserve the same API surface in a way that feels natural for each language.

Between releases 4.2 and 4.3-beta, the Java reference implementation saw significant changes:

$ git diff --stat 4.2..4.3-beta -- '*.java' | tail -1
  79 files changed, 4820 insertions(+), 4679 deletions(-)
Here's how I'd approach this with my manual workflow:

Open the changeset in Fork (my git client) and scan through all changed files
Plan the porting order based on the dependency graph: interfaces and enums first (they're usually independent), then try to port dependencies before the classes that use them, hoping to maintain some compilability
Pick a type to port in Java, open a side-by-side diff, check if the type already exists in the target runtime or needs creation from scratch
Port changes line-by-line, method-by-method to the target language
Watch the illusion of order crumble: the dependency graph is cyclic, so there's no perfect porting order that keeps everything compiling (note to self: it would be nice if we had an acyclic type dependency graph)
Can't test individual pieces because a skeletal animation system needs all its parts working in concert
Port everything blind, then face a wall of compilation errors and bugs introduced because my brain was fried after hours of human transpilation
This is especially fun when porting from Java to C, the language pair with the biggest type system and memory management mismatch.

What makes this tractable is that we maintain the same API surface across all runtime implementations. If there's a class Animation in Java, there's also a class Animation in C#, C++, and every other runtime, in a corresponding file. This one-to-one mapping exists for 99% of types. Sure, there are quirks like Java files containing dozens of inner classes, but the structural consistency is there.

Here's an example of one of the more math-heavy types, PhysicsConstraint:

Java (PhysicsConstraint.java)

package com.esotericsoftware.spine;

import static com.esotericsoftware.spine.utils.SpineUtils.*;

/** Stores the current pose for a physics constraint. A physics constraint applies physics to bones.
 * <p>
 * See <a href="https://esotericsoftware.com/spine-physics-constraints">Physics constraints</a> in the Spine User Guide. */
public class PhysicsConstraint extends Constraint<PhysicsConstraint, PhysicsConstraintData, PhysicsConstraintPose> {
    BonePose bone;

    boolean reset = true;
    float ux, uy, cx, cy, tx, ty;
    float xOffset, xLag, xVelocity;
    float yOffset, yLag, yVelocity;
    float rotateOffset, rotateLag, rotateVelocity;
    float scaleOffset, scaleLag, scaleVelocity;
    float remaining, lastTime;

    public PhysicsConstraint (PhysicsConstraintData data, Skeleton skeleton) {
        super(data, new PhysicsConstraintPose(), new PhysicsConstraintPose());
        if (skeleton == null) throw new IllegalArgumentException("skeleton cannot be null.");

        bone = skeleton.bones.items[data.bone.index].constrained;
    }

    public PhysicsConstraint copy (Skeleton skeleton) {
        var copy = new PhysicsConstraint(data, skeleton);
        copy.pose.set(pose);
        return copy;
    }

    public void reset (Skeleton skeleton) {
        remaining = 0;
        lastTime = skeleton.time;
        reset = true;
        xOffset = 0;
        xLag = 0;
        xVelocity = 0;
        yOffset = 0;
        yLag = 0;
        yVelocity = 0;
        rotateOffset = 0;
        rotateLag = 0;
        rotateVelocity = 0;
        scaleOffset = 0;
        scaleLag = 0;
        scaleVelocity = 0;
    }

    /** Translates the physics constraint so next {@link #update(Skeleton, Physics)} forces are applied as if the bone moved an
     * additional amount in world space. */
    public void translate (float x, float y) {
        ux -= x;
        uy -= y;
        cx -= x;
        cy -= y;
    }

    /** Rotates the physics constraint so next {@link #update(Skeleton, Physics)} forces are applied as if the bone rotated around
     * the specified point in world space. */
    public void rotate (float x, float y, float degrees) {
        float r = degrees * degRad, cos = cos(r), sin = sin(r);
        float dx = cx - x, dy = cy - y;
        translate(dx * cos - dy * sin - dx, dx * sin + dy * cos - dy);
    }

    /** Applies the constraint to the constrained bones. */
    public void update (Skeleton skeleton, Physics physics) {
        PhysicsConstraintPose p = applied;
        float mix = p.mix;
        if (mix == 0) return;

        boolean x = data.x > 0, y = data.y > 0, rotateOrShearX = data.rotate > 0 || data.shearX > 0, scaleX = data.scaleX > 0;
        BonePose bone = this.bone;
        float l = bone.bone.data.length, t = data.step, z = 0;

        switch (physics) {
        case none:
            return;
        case reset:
            reset(skeleton);
            // Fall through.
        case update:
            float delta = Math.max(skeleton.time - lastTime, 0), aa = remaining;
            remaining += delta;
            lastTime = skeleton.time;

            float bx = bone.worldX, by = bone.worldY;
            if (reset) {
                reset = false;
                ux = bx;
                uy = by;
            } else {
                float a = remaining, i = p.inertia, f = skeleton.data.referenceScale, d = -1, m = 0, e = 0, ax = 0, ay = 0,
                    qx = data.limit * delta, qy = qx * Math.abs(skeleton.scaleY);
                qx *= Math.abs(skeleton.scaleX);
                if (x || y) {
                    if (x) {
                        float u = (ux - bx) * i;
                        xOffset += u > qx ? qx : u < -qx ? -qx : u;
                        ux = bx;
                    }
                    if (y) {
                        float u = (uy - by) * i;
                        yOffset += u > qy ? qy : u < -qy ? -qy : u;
                        uy = by;
                    }
                    if (a >= t) {
                        float xs = xOffset, ys = yOffset;
                        d = (float)Math.pow(p.damping, 60 * t);
                        m = t * p.massInverse;
                        e = p.strength;
                        float w = f * p.wind, g = f * p.gravity;
                        ax = (w * skeleton.windX + g * skeleton.gravityX) * skeleton.scaleX;
                        ay = (w * skeleton.windY + g * skeleton.gravityY) * skeleton.scaleY;
                        do {
                            if (x) {
                                xVelocity += (ax - xOffset * e) * m;
                                xOffset += xVelocity * t;
                                xVelocity *= d;
                            }
                            if (y) {
                                yVelocity -= (ay + yOffset * e) * m;
                                yOffset += yVelocity * t;
                                yVelocity *= d;
                            }
                            a -= t;
                        } while (a >= t);
                        xLag = xOffset - xs;
                        yLag = yOffset - ys;
                    }
                    z = Math.max(0, 1 - a / t);
                    if (x) bone.worldX += (xOffset - xLag * z) * mix * data.x;
                    if (y) bone.worldY += (yOffset - yLag * z) * mix * data.y;
                }
                if (rotateOrShearX || scaleX) {
                    float ca = atan2(bone.c, bone.a), c, s, mr = 0, dx = cx - bone.worldX, dy = cy - bone.worldY;
                    if (dx > qx)
                        dx = qx;
                    else if (dx < -qx) //
                        dx = -qx;
                    if (dy > qy)
                        dy = qy;
                    else if (dy < -qy) //
                        dy = -qy;
                    if (rotateOrShearX) {
                        mr = (data.rotate + data.shearX) * mix;
                        z = rotateLag * Math.max(0, 1 - aa / t);
                        float r = atan2(dy + ty, dx + tx) - ca - (rotateOffset - z) * mr;
                        rotateOffset += (r - (float)Math.ceil(r * invPI2 - 0.5f) * PI2) * i;
                        r = (rotateOffset - z) * mr + ca;
                        c = cos(r);
                        s = sin(r);
                        if (scaleX) {
                            r = l * bone.getWorldScaleX();
                            if (r > 0) scaleOffset += (dx * c + dy * s) * i / r;
                        }
                    } else {
                        c = cos(ca);
                        s = sin(ca);
                        float r = l * bone.getWorldScaleX() - scaleLag * Math.max(0, 1 - aa / t);
                        if (r > 0) scaleOffset += (dx * c + dy * s) * i / r;
                    }
                    a = remaining;
                    if (a >= t) {
                        if (d == -1) {
                            d = (float)Math.pow(p.damping, 60 * t);
                            m = t * p.massInverse;
                            e = p.strength;
                            float w = f * p.wind, g = f * p.gravity;
                            ax = (w * skeleton.windX + g * skeleton.gravityX) * skeleton.scaleX;
                            ay = (w * skeleton.windY + g * skeleton.gravityY) * skeleton.scaleY;
                        }
                        float rs = rotateOffset, ss = scaleOffset, h = l / f;
                        while (true) {
                            a -= t;
                            if (scaleX) {
                                scaleVelocity += (ax * c - ay * s - scaleOffset * e) * m;
                                scaleOffset += scaleVelocity * t;
                                scaleVelocity *= d;
                            }
                            if (rotateOrShearX) {
                                rotateVelocity -= ((ax * s + ay * c) * h + rotateOffset * e) * m;
                                rotateOffset += rotateVelocity * t;
                                rotateVelocity *= d;
                                if (a < t) break;
                                float r = rotateOffset * mr + ca;
                                c = cos(r);
                                s = sin(r);
                            } else if (a < t) //
                                break;
                        }
                        rotateLag = rotateOffset - rs;
                        scaleLag = scaleOffset - ss;
                    }
                    z = Math.max(0, 1 - a / t);
                }
                remaining = a;
            }
            cx = bone.worldX;
            cy = bone.worldY;
            break;
        case pose:
            z = Math.max(0, 1 - remaining / t);
            if (x) bone.worldX += (xOffset - xLag * z) * mix * data.x;
            if (y) bone.worldY += (yOffset - yLag * z) * mix * data.y;
        }

        if (rotateOrShearX) {
            float o = (rotateOffset - rotateLag * z) * mix, s, c, a;
            if (data.shearX > 0) {
                float r = 0;
                if (data.rotate > 0) {
                    r = o * data.rotate;
                    s = sin(r);
                    c = cos(r);
                    a = bone.b;
                    bone.b = c * a - s * bone.d;
                    bone.d = s * a + c * bone.d;
                }
                r += o * data.shearX;
                s = sin(r);
                c = cos(r);
                a = bone.a;
                bone.a = c * a - s * bone.c;
                bone.c = s * a + c * bone.c;
            } else {
                o *= data.rotate;
                s = sin(o);
                c = cos(o);
                a = bone.a;
                bone.a = c * a - s * bone.c;
                bone.c = s * a + c * bone.c;
                a = bone.b;
                bone.b = c * a - s * bone.d;
                bone.d = s * a + c * bone.d;
            }
        }
        if (scaleX) {
            float s = 1 + (scaleOffset - scaleLag * z) * mix * data.scaleX;
            bone.a *= s;
            bone.c *= s;
        }
        if (physics != Physics.pose) {
            tx = l * bone.a;
            ty = l * bone.c;
        }
        bone.modifyWorld(skeleton.update);
    }

    void sort (Skeleton skeleton) {
        Bone bone = this.bone.bone;
        skeleton.sortBone(bone);
        skeleton.updateCache.add(this);
        skeleton.sortReset(bone.children);
        skeleton.constrained(bone);
    }

    boolean isSourceActive () {
        return bone.bone.active;
    }

    /** The bone constrained by this physics constraint. */
    public BonePose getBone () {
        return bone;
    }

    public void setBone (BonePose bone) {
        this.bone = bone;
    }
}
C++ Header (PhysicsConstraint.h)

#ifndef Spine_PhysicsConstraint_h
#define Spine_PhysicsConstraint_h

#include <spine/Constraint.h>
#include <spine/PhysicsConstraintData.h>
#include <spine/PhysicsConstraintPose.h>
#include <spine/BonePose.h>
#include <spine/Vector.h>

namespace spine {
    class Skeleton;
    class BonePose;
    class PhysicsConstraintPose;

    /// Stores the current pose for a physics constraint. A physics constraint applies physics to bones.
    ///
    /// See https://esotericsoftware.com/spine-physics-constraints Physics constraints in the Spine User Guide.
    class SP_API PhysicsConstraint : public ConstraintGeneric<PhysicsConstraint, PhysicsConstraintData, PhysicsConstraintPose> {
        friend class Skeleton;
        friend class PhysicsConstraintTimeline;
        friend class PhysicsConstraintInertiaTimeline;
        friend class PhysicsConstraintStrengthTimeline;
        friend class PhysicsConstraintDampingTimeline;
        friend class PhysicsConstraintMassTimeline;
        friend class PhysicsConstraintWindTimeline;
        friend class PhysicsConstraintGravityTimeline;
        friend class PhysicsConstraintMixTimeline;
        friend class PhysicsConstraintResetTimeline;

    public:
        RTTI_DECL

        PhysicsConstraint(PhysicsConstraintData& data, Skeleton& skeleton);

        void update(Skeleton& skeleton, Physics physics) override;
        void sort(Skeleton& skeleton) override;
        bool isSourceActive() override;
        PhysicsConstraint* copy(Skeleton& skeleton);

        void reset(Skeleton& skeleton);

        /// Translates the physics constraint so next update() forces are applied as if the bone moved an additional amount in world space.
        void translate(float x, float y);

        /// Rotates the physics constraint so next update() forces are applied as if the bone rotated around the specified point in world space.
        void rotate(float x, float y, float degrees);

        /// The bone constrained by this physics constraint.
        BonePose& getBone();
        void setBone(BonePose& bone);

    private:
        BonePose* _bone;

        bool _reset;
        float _ux, _uy, _cx, _cy, _tx, _ty;
        float _xOffset, _xLag, _xVelocity;
        float _yOffset, _yLag, _yVelocity;
        float _rotateOffset, _rotateLag, _rotateVelocity;
        float _scaleOffset, _scaleLag, _scaleVelocity;
        float _remaining, _lastTime;
    };
}

#endif /* Spine_PhysicsConstraint_h */
C++ Implementation (PhysicsConstraint.cpp)

#include <spine/PhysicsConstraint.h>
#include <spine/PhysicsConstraintData.h>
#include <spine/PhysicsConstraintPose.h>
#include <spine/BonePose.h>
#include <spine/Skeleton.h>
#include <spine/SkeletonData.h>
#include <spine/BoneData.h>
#include <spine/Bone.h>
#include <spine/MathUtil.h>

using namespace spine;

RTTI_IMPL(PhysicsConstraint, Constraint)

PhysicsConstraint::PhysicsConstraint(PhysicsConstraintData &data, Skeleton &skeleton) : ConstraintGeneric<PhysicsConstraint, PhysicsConstraintData, PhysicsConstraintPose>(data),
                                                                                        _reset(true), _ux(0), _uy(0), _cx(0), _cy(0), _tx(0), _ty(0),
                                                                                        _xOffset(0), _xLag(0), _xVelocity(0), _yOffset(0), _yLag(0), _yVelocity(0),
                                                                                        _rotateOffset(0), _rotateLag(0), _rotateVelocity(0), _scaleOffset(0), _scaleLag(0), _scaleVelocity(0),
                                                                                        _remaining(0), _lastTime(0) {

    _bone = &skeleton._bones[(size_t) data._bone->getIndex()]->_constrained;
}

PhysicsConstraint *PhysicsConstraint::copy(Skeleton &skeleton) {
    PhysicsConstraint *copy = new (__FILE__, __LINE__) PhysicsConstraint(_data, skeleton);
    copy->_pose.set(_pose);
    return copy;
}

void PhysicsConstraint::reset(Skeleton &skeleton) {
    _remaining = 0;
    _lastTime = skeleton.getTime();
    _reset = true;
    _xOffset = 0;
    _xLag = 0;
    _xVelocity = 0;
    _yOffset = 0;
    _yLag = 0;
    _yVelocity = 0;
    _rotateOffset = 0;
    _rotateLag = 0;
    _rotateVelocity = 0;
    _scaleOffset = 0;
    _scaleLag = 0;
    _scaleVelocity = 0;
}

void PhysicsConstraint::translate(float x, float y) {
    _ux -= x;
    _uy -= y;
    _cx -= x;
    _cy -= y;
}

void PhysicsConstraint::rotate(float x, float y, float degrees) {
    float r = degrees * MathUtil::Deg_Rad, cosVal = MathUtil::cos(r), sinVal = MathUtil::sin(r);
    float dx = _cx - x, dy = _cy - y;
    translate(dx * cosVal - dy * sinVal - dx, dx * sinVal + dy * cosVal - dy);
}

void PhysicsConstraint::update(Skeleton &skeleton, Physics physics) {
    PhysicsConstraintPose &p = *_applied;
    float mix = p._mix;
    if (mix == 0) return;

    bool x = _data._x > 0, y = _data._y > 0, rotateOrShearX = _data._rotate > 0 || _data._shearX > 0, scaleX = _data._scaleX > 0;
    BonePose *bone = _bone;
    float l = bone->_bone->_data.getLength(), t = _data._step, z = 0;

    switch (physics) {
        case Physics_None:
            return;
        case Physics_Reset:
            reset(skeleton);
            // Fall through.
        case Physics_Update: {
            float delta = MathUtil::max(skeleton._time - _lastTime, 0.0f), aa = _remaining;
            _remaining += delta;
            _lastTime = skeleton._time;

            float bx = bone->_worldX, by = bone->_worldY;
            if (_reset) {
                _reset = false;
                _ux = bx;
                _uy = by;
            } else {
                float a = _remaining, i = p._inertia, f = skeleton._data.getReferenceScale(), d = -1, m = 0, e = 0, ax = 0, ay = 0,
                      qx = _data._limit * delta, qy = qx * MathUtil::abs(skeleton.getScaleY());
                qx *= MathUtil::abs(skeleton._scaleX);
                if (x || y) {
                    if (x) {
                        float u = (_ux - bx) * i;
                        _xOffset += u > qx ? qx : u < -qx ? -qx
                                                          : u;
                        _ux = bx;
                    }
                    if (y) {
                        float u = (_uy - by) * i;
                        _yOffset += u > qy ? qy : u < -qy ? -qy
                                                          : u;
                        _uy = by;
                    }
                    if (a >= t) {
                        float xs = _xOffset, ys = _yOffset;
                        d = MathUtil::pow(p._damping, 60 * t);
                        m = t * p._massInverse;
                        e = p._strength;
                        float w = f * p._wind, g = f * p._gravity;
                        ax = (w * skeleton._windX + g * skeleton._gravityX) * skeleton._scaleX;
                        ay = (w * skeleton._windY + g * skeleton._gravityY) * skeleton.getScaleY();
                        do {
                            if (x) {
                                _xVelocity += (ax - _xOffset * e) * m;
                                _xOffset += _xVelocity * t;
                                _xVelocity *= d;
                            }
                            if (y) {
                                _yVelocity -= (ay + _yOffset * e) * m;
                                _yOffset += _yVelocity * t;
                                _yVelocity *= d;
                            }
                            a -= t;
                        } while (a >= t);
                        _xLag = _xOffset - xs;
                        _yLag = _yOffset - ys;
                    }
                    z = MathUtil::max(0.0f, 1 - a / t);
                    if (x) bone->_worldX += (_xOffset - _xLag * z) * mix * _data._x;
                    if (y) bone->_worldY += (_yOffset - _yLag * z) * mix * _data._y;
                }
                if (rotateOrShearX || scaleX) {
                    float ca = MathUtil::atan2(bone->_c, bone->_a), c, s, mr = 0, dx = _cx - bone->_worldX, dy = _cy - bone->_worldY;
                    if (dx > qx)
                        dx = qx;
                    else if (dx < -qx)
                        dx = -qx;
                    if (dy > qy)
                        dy = qy;
                    else if (dy < -qy)
                        dy = -qy;
                    if (rotateOrShearX) {
                        mr = (_data._rotate + _data._shearX) * mix;
                        z = _rotateLag * MathUtil::max(0.0f, 1 - aa / t);
                        float r = MathUtil::atan2(dy + _ty, dx + _tx) - ca - (_rotateOffset - z) * mr;
                        _rotateOffset += (r - MathUtil::ceil(r * MathUtil::InvPi_2 - 0.5f) * MathUtil::Pi_2) * i;
                        r = (_rotateOffset - z) * mr + ca;
                        c = MathUtil::cos(r);
                        s = MathUtil::sin(r);
                        if (scaleX) {
                            r = l * bone->getWorldScaleX();
                            if (r > 0) _scaleOffset += (dx * c + dy * s) * i / r;
                        }
                    } else {
                        c = MathUtil::cos(ca);
                        s = MathUtil::sin(ca);
                        float r = l * bone->getWorldScaleX() - _scaleLag * MathUtil::max(0.0f, 1 - aa / t);
                        if (r > 0) _scaleOffset += (dx * c + dy * s) * i / r;
                    }
                    a = _remaining;
                    if (a >= t) {
                        if (d == -1) {
                            d = MathUtil::pow(p._damping, 60 * t);
                            m = t * p._massInverse;
                            e = p._strength;
                            float w = f * p._wind, g = f * p._gravity;
                            ax = (w * skeleton._windX + g * skeleton._gravityX) * skeleton._scaleX;
                            ay = (w * skeleton._windY + g * skeleton._gravityY) * skeleton.getScaleY();
                        }
                        float rs = _rotateOffset, ss = _scaleOffset, h = l / f;
                        while (true) {
                            a -= t;
                            if (scaleX) {
                                _scaleVelocity += (ax * c - ay * s - _scaleOffset * e) * m;
                                _scaleOffset += _scaleVelocity * t;
                                _scaleVelocity *= d;
                            }
                            if (rotateOrShearX) {
                                _rotateVelocity -= ((ax * s + ay * c) * h + _rotateOffset * e) * m;
                                _rotateOffset += _rotateVelocity * t;
                                _rotateVelocity *= d;
                                if (a < t) break;
                                float r = _rotateOffset * mr + ca;
                                c = MathUtil::cos(r);
                                s = MathUtil::sin(r);
                            } else if (a < t)
                                break;
                        }
                        _rotateLag = _rotateOffset - rs;
                        _scaleLag = _scaleOffset - ss;
                    }
                    z = MathUtil::max(0.0f, 1 - a / t);
                }
                _remaining = a;
            }
            _cx = bone->_worldX;
            _cy = bone->_worldY;
            break;
        }
        case Physics_Pose: {
            z = MathUtil::max(0.0f, 1 - _remaining / t);
            if (x) bone->_worldX += (_xOffset - _xLag * z) * mix * _data._x;
            if (y) bone->_worldY += (_yOffset - _yLag * z) * mix * _data._y;
            break;
        }
    }

    if (rotateOrShearX) {
        float o = (_rotateOffset - _rotateLag * z) * mix, s, c, a;
        if (_data._shearX > 0) {
            float r = 0;
            if (_data._rotate > 0) {
                r = o * _data._rotate;
                s = MathUtil::sin(r);
                c = MathUtil::cos(r);
                a = bone->_b;
                bone->_b = c * a - s * bone->_d;
                bone->_d = s * a + c * bone->_d;
            }
            r += o * _data._shearX;
            s = MathUtil::sin(r);
            c = MathUtil::cos(r);
            a = bone->_a;
            bone->_a = c * a - s * bone->_c;
            bone->_c = s * a + c * bone->_c;
        } else {
            o *= _data._rotate;
            s = MathUtil::sin(o);
            c = MathUtil::cos(o);
            a = bone->_a;
            bone->_a = c * a - s * bone->_c;
            bone->_c = s * a + c * bone->_c;
            a = bone->_b;
            bone->_b = c * a - s * bone->_d;
            bone->_d = s * a + c * bone->_d;
        }
    }
    if (scaleX) {
        float s = 1 + (_scaleOffset - _scaleLag * z) * mix * _data._scaleX;
        bone->_a *= s;
        bone->_c *= s;
    }
    if (physics != Physics_Pose) {
        _tx = l * bone->_a;
        _ty = l * bone->_c;
    }
    bone->modifyWorld(skeleton._update);
}

void PhysicsConstraint::sort(Skeleton &skeleton) {
    Bone *bone = _bone->_bone;
    skeleton.sortBone(bone);
    skeleton._updateCache.add(this);
    skeleton.sortReset(bone->_children);
    skeleton.constrained(*bone);
}

bool PhysicsConstraint::isSourceActive() {
    return _bone->_bone->isActive();
}

BonePose &PhysicsConstraint::getBone() {
    return *_bone;
}

void PhysicsConstraint::setBone(BonePose &bone) {
    _bone = &bone;
}
Much of this porting work is mechanical and can be automated, like getters and setters, transferring documentation from Javadoc to docstrings, or ensuring the math matches. Some of the porting work requires a human brain, like translating Java generics to C++ templates, a task LLMs aren't very good at. What LLMs are good at is helping me double-check that I ported every line faithfully. This presented the perfect opportunity to apply my little workflow experiment to a real-world task on a real-world, largish codebase.

The "Port Java to X" Program
Time to write our program. We're essentially encoding my manual workflow into a structured LLM program. The goal: go through each Java type that changed between two commits and port it to a target runtime (like C++) collaboratively with the user.

Instead of me manually opening diffs, tracking dependencies, and porting line-by-line while my brain melts, we'll have the LLM handle the mechanical parts while I stay in control of the decisions that matter.

The final result of this program design can be found in the spine-port repository. The program itself is stored in a file called port.md. When I want to start or continue porting, I start Claude Code in the spine-port directory and tell it to read the port.md file in full and execute the workflow. That starts the "program".

Writing the port.md file was an iterative, collaborative process between me and Claude. In the following sections, we'll walk through each section of this program.

Initial Input and State
Every program needs input data and initial state to work with. The port.md file starts by defining the data structure that serves as both:

# Spine Runtimes Porting Program

Collaborative porting of changes between two commits in the Spine runtime
reference implementation (Java) to a target runtime. Work tracked in
`porting-plan.json` which has the following format:

​```json
{
  "metadata": {
    "prevBranch": "4.2",
    "currentBranch": "4.3-beta",
    "generated": "2024-06-30T...",
    "spineRuntimesDir": "/absolute/path/to/spine-runtimes",
    "targetRuntime": "spine-cpp",
    "targetRuntimePath": "/absolute/path/to/spine-runtimes/spine-cpp/spine-cpp",
    "targetRuntimeLanguage": "cpp"
  },
  "deletedFiles": [
    {
      "filePath": "/path/to/deleted/File.java",
      "status": "pending"
    }
  ],
  "portingOrder": [
    {
      "javaSourcePath": "/path/to/EnumFile.java",
      "types": [
        {
          "name": "Animation",
          "kind": "enum",
          "startLine": 45,
          "endLine": 52,
          "isInner": false,
          "portingState": "pending",
          "candidateFiles": ["/path/to/spine-cpp/include/spine/Animation.h", "/path/to/spine-cpp/include/spine/Animation.cpp"]
        }
      ]
    }
  ]
}
​```
This data structure is the central state that tracks our porting progress. The porting-plan.json file serves as both the initial input and the persistent state for our LLM program. Let's break down what each part means:

metadata: Configuration for the porting session:

prevBranch and currentBranch: The git commits we're porting between
spineRuntimesDir: Where all the runtime implementations live
targetRuntime, targetRuntimePath, targetRuntimeLanguage: Which runtime we're porting to and where to find it
deletedFiles: Java files that were removed and need corresponding deletions in the target runtime

portingOrder: The heart of the plan. Each entry represents a Java file that changed and contains:

javaSourcePath: The full path to the Java source file
types: An array of all classes, interfaces, and enums in that file, each with:
name: The type name (e.g., "Animation")
kind: Whether it's a class, interface, or enum
startLine and endLine: Where to find it in the Java file
isInner: Whether it's an inner type
portingState: Tracks progress: "pending" or "done"
candidateFiles: Where this type likely exists in the target runtime
The portingState field is crucial: it's how the LLM tracks what's been done across sessions. When I stop and restart later, the program knows exactly where to pick up.

But how do we generate all this structured data? Before the LLM can start porting, we need to analyze what changed between the two versions and prepare the data in a format the LLM can efficiently query. I wrote generate-porting-plan.js to automate this preparation:

./generate-porting-plan.js 4.2 4.3-beta /path/to/spine-runtimes spine-cpp
This script does several things:

Runs git diff to find all Java files that changed between the two commits
Uses lsp-cli to extract complete type information from both the Java reference implementation and the target runtime
Analyzes dependencies to create a porting order (enums before interfaces before classes)
Finds candidate files in the target runtime where each type likely exists
Outputs a structured JSON file that the LLM can read from and write to efficiently via jq.
Why pre-generate all this data instead of having the LLM explore the codebase as it goes?

Determinism: The same inputs always produce the same plan. LLM exploration might miss files, use wrong search patterns, or get lost in the directory structure. Pre-generation ensures complete, accurate, and reproducible results.

Context efficiency: The pre-generation saves tokens and turns by not needing the LLM to perform all the steps that generate-porting-plan.js does: running git diff, analyzing dependencies, extracting type information, finding candidate files. That's a lot of tool calls and context that would otherwise be wasted.

Speed: Generating the plan takes seconds. Having the LLM explore the codebase would take orders of magnitude longer.

This transforms an open-ended exploration problem into structured data processing. Instead of "figure out what changed and how to port it," the LLM gets "here's exactly what changed, where it lives, and where it should go."

No tokens and turns wasted. The context contains only the information we need.

The Function Library
The next section of port.md defines the tools available to our LLM program. Just like traditional programs import libraries, our LLM program needs tools to interact with the world.

## Tools

### VS Claude
Use the vs-claude MCP server tools for opening files and diffs for the user
during porting.

​```javascript
// Open multiple files at once (batch operations)
mcp__vs-claude__open([
  {"type": "file", "path": "/abs/path/Animation.java"},    // Java source
  {"type": "file", "path": "/abs/path/Animation.h"},       // C++ header
  {"type": "file", "path": "/abs/path/Animation.cpp"}      // C++ source
]);

// Open single file with line range
mcp__vs-claude__open({"type": "file", "path": "/abs/path/Animation.java", "startLine": 100, "endLine": 120});

// View git diff for a file
mcp__vs-claude__open({"type": "gitDiff", "path": "/abs/path/Animation.cpp", "from": "HEAD", "to": "working"});
​```
VS Code Integration keeps the human in the loop. The LLM doesn't just modify files in the background; it opens them in my editor so I can see what's happening. This uses vs-claude, an extension I wrote that implements the Model Context Protocol (MCP), turning VS Code into the LLM's display device.

### Progress Tracking

Monitor porting progress using these jq commands:

​```bash
# Get overall progress percentage
jq -r '.portingOrder | map(.types[]) | "\(([.[] | select(.portingState == "done")] | length)) types ported out of \(length) total (\(([.[] | select(.portingState == "done")] | length) * 100 / length | floor)% complete)"' porting-plan.json

# Count types by state
jq -r '.portingOrder | map(.types[]) | group_by(.portingState) | map({state: .[0].portingState, count: length}) | sort_by(.state)' porting-plan.json

# List all completed types
jq -r '.portingOrder | map(.types[] | select(.portingState == "done") | .name) | sort | join(", ")' porting-plan.json

# Find remaining types to port
jq -r '.portingOrder | map(.types[] | select(.portingState == "pending") | .name) | length' porting-plan.json
​```
Progress Tracking functions that I can ask the LLM to run at any time. When I want to know how far we've gotten, I tell the LLM to execute one of these queries. They're pre-written and tested, ensuring the LLM uses efficient patterns instead of crafting its own potentially buggy versions.

### Compile Testing

For C++, test compile individual files during porting:

​```bash
./compile-cpp.js /path/to/spine-cpp/spine-cpp/src/spine/Animation.cpp
​```

For other languages, we can not compile individual files and should not try to.
Compile Testing includes language-specific build commands. Notice the explicit warning about other languages. This prevents the LLM from wasting time trying to compile individual TypeScript or Haxe files, which would fail. It's defensive programming: explicitly state what not to do. Though with our non-deterministic computer, there's always a chance it'll try anyway.

The Main Workflow
Now we get to the heart of our program: the actual porting logic with its loops, conditionals, and human checkpoints.

## Workflow

Port one type at a time. Ensure the target runtime implementation is functionally
equivalent to the reference implementation. The APIs must match, bar idiomatic
differences, including type names, field names, method names, enum names,
parameter names and so on. Implementations of methods must match EXACTLY, bar
idiomatic differences, such as differences in collection types.

Follow these steps to port each type:

### 1. Setup (One-time)

DO NOT use the TodoWrite and TodoRead tools for this phase!

1. Read metadata from porting-plan.json:
   ​```bash
   jq '.metadata' porting-plan.json
   ​```
   - If this fails, abort and tell user to run generate-porting-plan.js
   - Store these values for later use:
      - targetRuntime (e.g., "spine-cpp")
      - targetRuntimePath (e.g., "/path/to/spine-cpp/spine-cpp")
      - targetRuntimeLanguage (e.g., "cpp")
Like any traditional program, this starts with initialization. The jq '.metadata' command is essentially a function call - the program tells the LLM exactly which function to invoke rather than having it figure out the extraction logic itself, saving tokens and turns. Notice the conditional: "If this fails, abort" - defensive programming ensuring the required input exists. The "Store these values" instruction doesn't actually store anything; it tells the LLM what variable names to use when referencing these values extracted from the porting plan metadata later, similar to how variable assignments work. Storage happens implicitly in the LLM's context.

2. In parallel
   a. Check for conventions file:
      - Read `${targetRuntime}-conventions.md` (from step 1) in full.
      - If missing:
         - Use Task agents in parallel to analyze targetRuntimePath (from step 1)
         - Document all coding patterns and conventions:
            * Class/interface/enum definition syntax
            * Member variable naming (prefixes like m_, _, etc.)
            * Method naming conventions (camelCase vs snake_case)
            * Inheritance syntax
            * File organization (single file vs header/implementation)
            * Namespace/module/package structure
            * Memory management (GC, manual, smart pointers)
            * Error handling (exceptions, error codes, Result types)
            * Documentation format (Doxygen, JSDoc, etc.)
            * Type system specifics (generics, templates)
            * Property/getter/setter patterns
      - Agents MUST use ripgrep instead of grep!
      - Save as ${TARGET}-conventions.md
      - STOP and ask the user to review the generated conventions file

   b. Read `porting-notes.md` in full
      - If missing create with content:
      ​```markdown
      # Porting Notes
      ​```
The "in parallel" instruction tells the LLM to use multiple tools simultaneously rather than sequentially, speeding up execution.

Step 2a first tries to read the conventions file using ${targetRuntime}-conventions.md - this is like string interpolation, where the LLM inserts the value for the target runtime name that it read from the porting plan. That conventions file captures the coding style and architecture patterns of the target runtime. This documents everything from naming conventions to memory management approaches, ensuring the LLM ports code that fits seamlessly into the existing codebase.

Step 2b sets up the porting notes file as a scratch pad for the porting job. This contains observations and edge cases discovered while porting - things like "Java uses pose but C++ uses _applied in this context" or "don't port toString() methods." It's a running log so we don't forget important discoveries that affect later porting decisions.

If either file doesn't exist, we create it.

Why generate the conventions file as part of the workflow instead of pre-generating it like the porting plan? Pre-generating conventions would require either manual work (time-consuming) or using an LLM with manual review anyway. Generating it inside the program ensures I don't have too many moving parts to juggle, and all instructions aimed at an LLM are contained in a single location: the program.

Here are the actual conventions and porting notes files:

Conventions (spine-cpp-conventions.md)

# Spine-CPP Coding Conventions

## File Organization

### Directory Structure
- Headers: `include/spine/` - Public API headers
- Implementation: `src/spine/` - Implementation files (.cpp)
- One class per file pattern (Animation.h/Animation.cpp)

### Header Guards
​```cpp
#ifndef Spine_Animation_h
#define Spine_Animation_h
// content
#endif
​```

## Naming Conventions

### Classes and Types
- **Classes**: PascalCase (e.g., `Animation`, `BoneTimeline`, `SkeletonData`)
- **Enums**: PascalCase with prefixed values
  ​```cpp
  enum MixBlend {
      MixBlend_Setup,
      MixBlend_First,
      MixBlend_Replace,
      MixBlend_Add
  };
  ​```

### Variables
- **Member variables**: Underscore prefix + camelCase
  ​```cpp
  private:
      float _duration;
      Vector<Timeline*> _timelines;
      bool _hasDrawOrder;
  ​```
- **Parameters**: camelCase, often with "inValue" suffix for setters
  ​```cpp
  void setDuration(float inValue) { _duration = inValue; }
  ​```
- **Local variables**: camelCase (e.g., `timelineCount`, `lastTime`)

### Methods
- **Public methods**: camelCase
  ​```cpp
  float getDuration() { return _duration; }
  void apply(Skeleton& skeleton, float lastTime, float time, ...);
  ​```
- **Getters/Setters**: get/set prefix pattern
  ​```cpp
  const String& getName() { return _name; }
  void setName(const String& inValue) { _name = inValue; }
  ​```

## Class Structure

### Base Class Pattern
All classes inherit from `SpineObject`:
​```cpp
class SP_API Animation : public SpineObject {
    // ...
};
​```

### Access Modifiers Order
1. Friend declarations
2. Public members
3. Protected members
4. Private members

### Friend Classes
Extensive use for internal access:
​```cpp
friend class AnimationState;
friend class AnimationStateData;
​```

## Memory Management

### Manual Memory Management
- No smart pointers
- Explicit new/delete with custom allocators
- SpineObject base class provides memory tracking

### Object Pools
​```cpp
class SP_API Pool<T> : public SpineObject {
    // Efficient object recycling
};
​```

### Vector Usage
Custom Vector class instead of std::vector:
​```cpp
Vector<Timeline*> _timelines;
​```

## Type System

### Custom String Class
​```cpp
class SP_API String : public SpineObject {
    // UTF-8 aware string implementation
};
​```

### Container Types
- `Vector<T>` - Dynamic array
- `HashMap<K,V>` - Hash map implementation
- `Pool<T>` - Object pool

### RTTI Usage
Runtime type information with RTTI macros:
​```cpp
RTTI_DECL
RTTI_IMPL(Animation, SpineObject)
​```

## Error Handling

### No Exceptions
- Uses assertions for debug builds
- Return values indicate success/failure
- assert() for internal invariants

### Assertions
​```cpp
assert(timelineCount == _timelines.size());
​```

## Documentation

### Triple-Slash Comments
​```cpp
/// @brief Sets the duration of the animation.
/// @param inValue The duration in seconds.
void setDuration(float inValue);
​```

### File Headers
​```cpp
/******************************************************************************
 * Spine Runtimes License Agreement
 * ...
 *****************************************************************************/
​```

## Platform Handling

### DLL Export/Import
​```cpp
#ifdef SPINE_CPP_DLL
    #ifdef _MSC_VER
        #define SP_API __declspec(dllexport)
    #else
        #define SP_API __attribute__((visibility("default")))
    #endif
#else
    #define SP_API
#endif
​```

### Platform-Specific Code
Minimal, mostly in Extension.cpp

## Include Style

### System Headers
​```cpp
#include <spine/SpineObject.h>
#include <spine/Vector.h>
​```

### Forward Declarations
Used extensively to minimize dependencies:
​```cpp
namespace spine {
    class Skeleton;
    class Event;
}
​```

## Special Patterns

### Inline Getters/Setters
Simple accessors defined in headers:
​```cpp
float getDuration() { return _duration; }
​```

### Const Correctness
- Const methods where appropriate
- Const references for parameters
​```cpp
void apply(Skeleton& skeleton, float lastTime, float time,
          Vector<Event*>* pEvents, float alpha,
          MixBlend blend, MixDirection direction);
​```

### No STL Usage
Custom implementations for all containers, minimal stdlib dependency

## Build System

### CMake Integration
- include directories specified in CMakeLists.txt
- Separate static/shared library targets

### Compiler Flags
- C++11 standard
- RTTI enabled
- No exceptions (-fno-exceptions on some platforms)
Porting Notes (porting-notes.md)

# Porting Notes

## Java to C++ Porting Conventions

### Direct Field Access vs Getters/Setters
- C++ uses direct field access (e.g., `p._mixRotate`) where Java uses direct field access (e.g., `p.mixRotate`)
- C++ uses underscore prefix for private fields but accesses them directly via friend classes
- This is intentional - C++ classes are friends of each other to allow direct field access
- Do NOT change direct field access to getter/setter calls in C++
- Java `slot.pose.attachment` → C++ `_slot->_pose._attachment` (NOT `_slot->getPose()._attachment`)

### Override Keywords
- When fixing override warnings, add `virtual` and `override` keywords to methods in derived classes
- Example: `PathConstraintData &getData();` → `virtual PathConstraintData &getData() override;`

### Pose vs Applied Usage (CRITICAL)
- C++ MUST use the same pose reference as Java
- If Java uses `applied`, C++ must use `_applied`
- If Java uses `pose`, C++ must use `_pose`
- Never mix these up - they represent different states in the constraint system
- Example:
  - Java: `IkConstraintPose p = applied;` → C++: `IkConstraintPose &p = *_applied;`
  - Java: `copy.pose.set(pose);` → C++: `copy->_pose.set(_pose);`

### Constraint Pose Initialization
- In Java, constraint constructors pass new pose instances to the parent constructor
- In C++, `ConstraintGeneric` allocates poses as member fields (_pose and _constrained), not through constructor
- Example: Java `super(data, new PhysicsConstraintPose(), new PhysicsConstraintPose())`
- C++ just calls `ConstraintGeneric<PhysicsConstraint, PhysicsConstraintData, PhysicsConstraintPose>(data)`

### Method Name Differences
- Java `Posed.reset()` → C++ `PosedGeneric::resetConstrained()`
- This naming difference must exist to avoid conflicts with other reset methods in C++
- The Java code has a comment `// Port: resetConstrained` indicating this is intentional

### Methods NOT Ported
- `toString()` methods are not ported to C++
- Methods that use `Matrix3` type in Java are not ported to C++

### Parameter Conversions
- Methods that take `Vector2` in Java are "unrolled" in C++ to use separate float parameters:
  - Java: `worldToLocal(Vector2 world)`
  - C++: `worldToLocal(float worldX, float worldY, float& outLocalX, float& outLocalY)`

## Dark Color Handling Pattern
- Java uses `@Null Color darkColor` (nullable reference)
- C++ uses `Color _darkColor` with `bool _hasDarkColor` flag
- This avoids manual memory management while maintaining the same functionality
- The _hasDarkColor boolean indicates whether a dark color has been set
- Example: SlotPose class uses this pattern

## Event Class (Ported - 4.3-beta update)
- Method names updated to match Java API:
  - `getIntValue()` → `getInt()`
  - `setIntValue()` → `setInt()`
  - `getFloatValue()` → `getFloat()`
  - `setFloatValue()` → `setFloat()`
  - `getStringValue()` → `getString()`
  - `setStringValue()` → `setString()`
- Constructor now initializes values from EventData defaults to match Java behavior
At this point, we have exactly what we need in the context and nothing else: port.md in full (the program), the metadata from the porting plan JSON, the target runtime conventions in full, and the porting notes in full.

And now here's the main loop that does the actual porting:

### 2. Port Types (Repeat for each)

1. **Find next pending type:**
   ​```bash
   # Get next pending type info with candidate files
   jq -r '.portingOrder[] | {file: .javaSourcePath, types: .types[] | select(.portingState == "pending")} | "\(.file)|\(.types.name)|\(.types.kind)|\(.types.startLine)|\(.types.endLine)|\(.types.candidateFiles | join(","))"' porting-plan.json | head -1
   ​```

2. **Open files in VS Code via vs-claude (for user review):**
   - Open Java file and Java file git diff (from prevBranch to currentBranch)
   - If candidateFiles exists: open all candidate files

3. **Confirm with user:**
   - Ask: "Port this type? (y/n)"
   - STOP and wait for confirmation.

4. **Read source files:**
   - Note: Read the files in parallel if possible
   - Java: Read the ENTIRE file so it is fully in your context!
   - Target: If exists, read the ENTIRE file(s) so they are fully in your context!
   - For large files (>2000 lines): Read in chunks of 1000 lines
   - Read parent types if needed (check extends/implements)
   - Goal: Have complete files in context for accurate porting

5. **Port the type:**
   - Follow conventions from ${targetRuntime}-conventions.md
   - If target file(s) don't exist, create them and open them for the user via vs-claude
   - Port incrementally and always ultrathink:
     * Base on the full content of the files in your context, identify differences and changes that need to be made.
      * differences can be due to idiomatic differences, or real differences due to new or changed functionality in the reference
        implementation. Ultrathink to discern which is which.
     * If changes need to be made:
       * Structure first (fields, method signatures)
       * Then method implementations
       * For C++: Run `./compile-cpp.js` after each method
   - Use MultiEdit for all changes to one file
   - Ensure 100% functional parity
   - Add or update jsdoc, doxygen, etc. based on Javadocs.

6. **Get user confirmation:**
   - Open a diff of the files you modified, comparing HEAD to working.
   - Give the user a summary of what you ported
   - Ask: "Mark as done? (y/n)"
   - If yes, update status:
   ​```bash
   jq --arg file "path/to/file.java" --arg type "TypeName" \
      '(.portingOrder[] | select(.javaSourcePath == $file) | .types[] | select(.name == $type) | .portingState) = "done"' \
      porting-plan.json > tmp.json && mv tmp.json porting-plan.json
   ​```

7. **Update porting-notes.md:**
   - Add any new patterns or special cases discovered.

8. **STOP and confirm:**
   - Show what was ported. Ask: "Continue to next type? (y/n)"
   - Only proceed after confirmation.
Step 1 queries our state to find the next pending type using another precise jq function call. Here's what gets extracted from the porting plan and is now in the context:

{
  "javaSourcePath": "/Users/badlogic/workspaces/spine-runtimes/spine-libgdx/spine-libgdx/src/com/esotericsoftware/spine/attachments/AtlasAttachmentLoader.java",
  "types": [
    {
      "name": "AtlasAttachmentLoader",
      "kind": "class",
      "startLine": 43,
      "endLine": 101,
      "isInner": false,
      "portingState": "pending",
      "candidateFiles": [
        "/Users/badlogic/workspaces/spine-runtimes/spine-cpp/spine-cpp/include/spine/AtlasAttachmentLoader.h",
        "/Users/badlogic/workspaces/spine-runtimes/spine-cpp/spine-cpp/src/spine/AtlasAttachmentLoader.cpp"
      ]
    }
  ]
}
Steps 2-3 implement a human checkpoint: the LLM opens relevant files in VS Code and waits for confirmation before proceeding. This isn't just politeness; it's a safety mechanism ensuring I can abort if something looks wrong, or if I want to port a different type, or if I can immediately see that this type is complete and we should update the porting state and move on to the next type.

Step 4 is critical: "Read the ENTIRE file so it is fully in your context!" The emphasis prevents the LLM from being lazy and reading only parts of files, which would lead to incomplete ports. After this step, the context contains the complete source files needed for accurate porting.

Step 5 does the actual porting work. The "ultrathink to discern" instruction acknowledges that porting isn't just mechanical translation; it requires judgment about idiomatic differences versus actual functionality changes.

Steps 6-8 implement more human checkpoints and program state management. The LLM shows its work (via diffs), then updates the program's persistent state: step 6 writes to porting-plan.json to mark the type as done, and step 7 optionally updates porting-notes.md with new observations. This is where our program writes its state to disk - crucial for resumability. The state update in step 6 uses jq to surgically modify just the portingState field, ensuring we can resume exactly where we left off. Step 8 asks permission before the next loop iteration.

This is our complete LLM program: structured initialization, precise function calls, state management, human checkpoints, and deterministic loops. What started as an ad hoc conversation with an AI has become a reproducible, resumable workflow that tackles real-world porting at scale.

Program Execution
Let's see this program in action:

This recording demonstrates the key insights of treating LLMs as computers:

Follows the program: Reads port.md and executes each step methodically
Manages state: Uses precise jq queries to read and update porting-plan.json
Loads context strategically: Reads only necessary files when needed
Maintains resumability: Updates persistent state after each completed type
Implements human checkpoints: Asks for confirmation before proceeding
Unlike ad hoc prompting where the conversation meanders, this programmatic approach follows a deterministic workflow. The LLM becomes a reliable executor of structured instructions rather than an unpredictable chat partner.

While the LLM cannot handle the full porting process autonomously due to challenges with complex type hierarchies and execution flows, this deterministic workflow dramatically simplifies my work compared to manual porting. The tedious mechanical tasks are now automated, freeing me to focus on the genuinely difficult problems that require human insight.

What previously took me 2-3 weeks, now takes 2-3 days.

Future Work
As with everything, there's always plenty of room for improvements.

Testing and debugging: The deterministic nature of these workflows opens possibilities for traditional software engineering practices. We could test that given specific inputs and state, our programs produce expected outputs. For debugging, we could instrument prompts to write state information at key points, creating execution traces that help identify where workflows diverge from expectations.

Sub-agent orchestration: Current tools like Claude Code spawn sub-agents without observability or communication channels. To scale this programming model, we need structured ways to program sub-agents and monitor their execution. The main agent should define explicit workflows for sub-agents rather than generating them ad hoc, ensuring reliability through the same structured approach we use for the primary workflow.

Conclusion
This mental model has transformed how I work with established codebases using agentic coding tools. By treating LLMs as programmable computers rather than conversational partners, I've found a more reliable approach to complex software tasks. It's not a panacea and won't work for all problems, but it represents a step toward turning AI-assisted coding into an engineering discipline rather than a "throwing shit at the wall" approach.