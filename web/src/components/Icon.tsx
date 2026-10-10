// Inline Lucide icons (24 viewBox, stroke 1.75, round caps), exported from the design into src/assets/icons.
const RAW = import.meta.glob('../assets/icons/*.svg', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

export type IconName = 'sun' | 'volume-2' | 'mic' | 'pause' | 'check' | 'leaf' | 'redo-2' | 'x' | 'shield' | 'bell';

function inner(name: IconName): string {
  const src = RAW[`../assets/icons/${name}.svg`] ?? '';
  return src.replace(/^[\s\S]*?<svg[^>]*>/, '').replace(/<\/svg>\s*$/, '');
}

type Props = { name: IconName; size: number; stroke?: string; strokeWidth?: number; style?: React.CSSProperties };

export function Icon({ name, size, stroke = 'currentColor', strokeWidth = 1.75, style }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke}
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={style}
      aria-hidden="true"
      dangerouslySetInnerHTML={{ __html: inner(name) }}
    />
  );
}
