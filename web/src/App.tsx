import { lazy, Suspense } from 'react';

// Fixture frame host: only reachable in dev/test builds (`/?frame=<id>`); tree-shaken out of production builds.
const FrameHost = import.meta.env.DEV ? lazy(() => import('./dev/FrameHost.tsx')) : null;

export function App() {
  const frame = typeof location === 'undefined' ? null : new URLSearchParams(location.search).get('frame');
  if (FrameHost && frame) {
    return (
      <Suspense fallback={null}>
        <FrameHost frame={frame} />
      </Suspense>
    );
  }
  // Production routing arrives with the behaviour wiring.
  return null;
}
