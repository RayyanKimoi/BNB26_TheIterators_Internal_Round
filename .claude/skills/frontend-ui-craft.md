---
name: frontend-ui-craft
description: Guidelines for building ultra-high-craft, performant, and scalable frontend components with smooth animations and zero visual slop.
---

# Frontend UI Craft & Component Design

## Core Architecture Rules
- **Proportional & Scaled Layouts**: Every card, grid item, and container must use responsive aspect ratios (`aspect-video`, `aspect-square`, or explicit flex/grid tracks) to prevent layout shifts or stretched cards.
- **Performant Animations**: Restrict scroll and hover animations to GPU-accelerated properties (`transform`, `opacity`, `filter`). Avoid animating layout properties (`width`, `height`, `margin`, `padding`).
- **Clean Component Scaffolding**: Keep canvas shaders, ASCII grids, and complex mathematical sliders isolated in modular sub-components inside `src/components/ui/` rather than cluttering core view files.
- **Zero Bloat Principle**: Implement visual effects using clean, native React + Framer Motion + Canvas APIs instead of installing heavy external UI libraries.

## Animation & Motion Standards
- **Framer Motion Setup**: Use `layoutId` for smooth tab transitions, `<AnimatePresence>` for mounted/unmounted elements, and `useScroll` / `useTransform` for scroll-linked effects.
- **Scroll Polish**: Wrap scrollable timelines and stack cards in containers with visible touch/mouse scroll indicators, `scroll-snap-type: y mandatory`, and smooth deceleration.
- **Reduced Motion**: Always check `useReducedMotion()` to fallback gracefully to standard static layouts for accessibility.

## Project-specific craft notes (from the UI/UX Pro Max skill)
- Scroll-triggered storytelling: narrative must read without the effects; keep DOM order complete; disable parallax/scrub under reduced motion; pause offscreen; show a progress indicator.
- Scroll reveal: 400-600ms, ease-out; do not stagger more than ~8 children.
- Parallax: vary speed per layer, clip overflow on the wrapper, batch layers under one scroll container.
- Honour this repo's hard rules first (CLAUDE.md): no purple gradients, no emoji icons, no fake metrics, no em dashes in UI copy, Geist Mono for all data, real numbers only.
