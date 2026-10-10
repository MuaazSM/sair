import { FRAMES } from './frames.tsx';

const VIEWPORTS: Record<string, { width: number; height: number }> = {};
const MOBILE = { width: 390, height: 844 };

/** Dev/test only. Renders one screen with its fixture inside a fixed frame-sized box (`/?frame=<id>`). */
export default function FrameHost({ frame }: { frame: string }) {
  const render = FRAMES[frame];
  const vp = VIEWPORTS[frame] ?? MOBILE;
  if (!render) return <pre>Unknown frame: {frame}</pre>;
  return (
    <div data-frame-root={frame} style={{ width: vp.width, height: vp.height, overflow: 'hidden' }}>
      {render()}
    </div>
  );
}
