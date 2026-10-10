import type { CSSProperties } from 'react';
import logoUrl from '../assets/sair-icon.svg';

/** The Sair mark. `alt=""` marks it decorative (design uses that inside illustrations). */
export function Logo({ size = 44, alt = 'Sair', style }: { size?: number; alt?: string; style?: CSSProperties }) {
  return <img src={logoUrl} alt={alt} style={{ width: size, height: size, ...style }} />;
}
