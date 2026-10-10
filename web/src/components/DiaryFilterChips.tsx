type Props = { labels: string[]; activeIndex: number };

/** Diary filter chips (design 3A: All / Mom / Dad). Labels come from props (group names are config, not copy).
 *  Static for now: renders the pressed state, no filtering behaviour. */
export function DiaryFilterChips({ labels, activeIndex }: Props) {
  return (
    <div style={{ display: 'flex', gap: 10 }}>
      {labels.map((label, i) => {
        const active = i === activeIndex;
        return (
          <button
            key={label}
            type="button"
            aria-pressed={active}
            style={{
              minHeight: 48,
              padding: '0 22px',
              borderRadius: 999,
              border: `1.5px solid ${active ? '#3F6B4A' : '#D9E0D6'}`,
              background: active ? '#3F6B4A' : '#FFFFFF',
              color: active ? '#FAFAF6' : '#2C4A34',
              font: "700 20px 'Nunito'",
              cursor: 'pointer',
              transition: 'background-color 300ms cubic-bezier(.22,.61,.36,1),color 300ms cubic-bezier(.22,.61,.36,1)',
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
