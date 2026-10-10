// Live app shell: a small state machine over the replicated screens (PRD §5 J1/J3, §6.2 FR-J1/J8/J9, §6.3 FR-D1-D4).
// No router library; the only routes are the hash routes `#setup?token=…` and `#diary`.
// Privacy: no console output, no analytics, no third-party requests. Mission/entry text lives in React state only.
import { useCallback, useEffect, useRef, useState } from 'react';
import { GapState } from '../components/index.ts';
import { door, doorFirstWalk, leavePhone, missionPlaying, recording, recordingWarning, saved as savedFixture, tellSair, writing } from '../fixtures/index.ts';
import type { DiaryData } from '../fixtures/diary.ts';
import type { SavedData } from '../fixtures/saved.ts';
import { Diary } from '../screens/Diary/index.ts';
import { Door } from '../screens/Door/index.ts';
import { DoorFirstWalk } from '../screens/DoorFirstWalk/index.ts';
import { LeavePhone } from '../screens/LeavePhone/index.ts';
import { MissionPlaying } from '../screens/MissionPlaying/index.ts';
import { Recording } from '../screens/Recording/index.ts';
import { Saved } from '../screens/Saved/index.ts';
import { TellSair } from '../screens/TellSair/index.ts';
import { Writing } from '../screens/Writing/index.ts';
import { getAudioBlobUrl, getDiary, getMission, getToken, getWalk, hasWalked, isTransient, markWalked, postWalk, setToken, type DiaryDto, type WalkStatusDto } from './api.ts';
import { clock, longDate, monthName, parseIsoDate, shortDate } from './format.ts';
import { MissionPlayer, type PlayState } from './player.ts';
import { addPending, deletePending, listPending } from './queue.ts';
import { MAX_SECONDS, Recorder, WARN_AT_SECONDS } from './recorder.ts';

type View =
  | { k: 'setup' }
  | { k: 'door' }
  | { k: 'playing' }
  | { k: 'leave'; dimmed: boolean }
  | { k: 'tell' }
  | { k: 'recording' }
  | { k: 'sending' }
  | { k: 'writing' }
  | { k: 'saved'; data: SavedData }
  | { k: 'slow' }
  | { k: 'sleeping'; seconds: number }
  | { k: 'couldnt' }
  | { k: 'lost' }
  | { k: 'mic' }
  | { k: 'unreachable' }
  | { k: 'diary' };

const SLOW_AFTER_MS = 45_000;
const POLL_MS = 2_500;
const POLL_LIMIT = 240; // 10 minutes
const BACKOFF_MS = [5_000, 10_000, 20_000, 40_000, 60_000];

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function bootstrap(): View {
  const h = location.hash;
  if (h.startsWith('#setup')) {
    const token = new URLSearchParams(h.split('?')[1] ?? '').get('token');
    if (token) setToken(token);
    history.replaceState(null, '', location.pathname + location.search); // the token leaves the address bar at once
  }
  if (!getToken()) return { k: 'setup' };
  return location.hash === '#diary' ? { k: 'diary' } : { k: 'door' };
}

function toSaved(w: WalkStatusDto, group: string): SavedData {
  const facts = (w.facts ?? []).map((f) => ({ kind: f.certainty === 'sure' ? ('sure' as const) : ('maybe' as const), text: f.place ? `${f.subject} · ${f.place}` : f.subject }));
  const outcome = w.mission_result === 'done' ? ({ kind: 'done', label: 'Mission done' } as const) : facts.length === 0 ? ({ kind: 'quiet', label: 'A quiet walk' } as const) : undefined;
  return { ...savedFixture, date: longDate(parseIsoDate(w.date)), group, entry: w.entry ?? '', facts, outcome };
}

function toDiaryData(d: DiaryDto, filters: string[], active: number): DiaryData {
  const n = d.summary.walks_this_month;
  const m = d.summary.things_noticed;
  return {
    title: 'Our walks',
    // NOT IN DESIGN: singular forms (the design only shows plurals).
    summary: `${n} ${n === 1 ? 'walk' : 'walks'} this month · ${m} ${m === 1 ? 'thing' : 'things'} noticed`,
    filters,
    activeFilter: active,
    entries: d.entries.map((e) => {
      const again = e.facts.flatMap((f) => f.asked_again_on).sort();
      const last = again[again.length - 1];
      return {
        date: shortDate(parseIsoDate(e.date)),
        group: e.group,
        mission: e.mission_text ?? '',
        entry: e.entry,
        facts: e.facts.map((f) => ({ kind: f.certainty === 'sure' ? ('sure' as const) : ('maybe' as const), text: f.place ? `${f.subject} · ${f.place}` : f.subject })),
        askedAgain: last ? `Asked about again on ${shortDate(parseIsoDate(last))}` : undefined,
      };
    }),
    footer: `That's everything from ${monthName(new Date())} so far.`,
  };
}

type Press = { released: boolean; recorder?: Recorder; tick?: ReturnType<typeof setInterval> };

export function LiveApp() {
  const [view, setView] = useState<View>(bootstrap);
  const viewRef = useRef<View>(view);
  const go = useCallback((v: View) => {
    viewRef.current = v;
    setView(v);
  }, []);

  const [missionText, setMissionText] = useState('');
  const [play, setPlay] = useState<PlayState>({ label: '', progress: 0, revealed: 0, paused: false });
  const [secs, setSecs] = useState(0);
  const [diary, setDiary] = useState<{ data: DiaryData | null; groups: string[]; filter: number }>({ data: null, groups: [], filter: 0 });

  const player = useRef<MissionPlayer | null>(null);
  const audioUrl = useRef<string | null>(null);
  const press = useRef<Press | null>(null);
  const poll = useRef<{ cancelled: boolean } | null>(null);
  const slowTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const attempt = useRef(0);
  const draining = useRef(false);
  const busy = useRef(false);

  // ---------------------------------------------------------------- polling (J3 steps 3-4)
  const stopPolling = useCallback(() => {
    if (poll.current) poll.current.cancelled = true;
    clearTimeout(slowTimer.current);
  }, []);

  const startPolling = useCallback(
    (walkId: string) => {
      stopPolling();
      const token = { cancelled: false };
      poll.current = token;
      go({ k: 'writing' });
      slowTimer.current = setTimeout(() => {
        if (!token.cancelled && viewRef.current.k === 'writing') go({ k: 'slow' });
      }, SLOW_AFTER_MS);
      void (async () => {
        for (let i = 0; i < POLL_LIMIT && !token.cancelled; i++) {
          await sleep(POLL_MS);
          if (token.cancelled) return;
          let w: WalkStatusDto;
          try {
            w = await getWalk(walkId);
          } catch {
            continue; // hub busy or asleep: keep waiting
          }
          if (w.status === 'received') continue;
          token.cancelled = true;
          clearTimeout(slowTimer.current);
          const waiting = viewRef.current.k === 'writing' || viewRef.current.k === 'slow';
          if (w.status === 'lost') {
            if (waiting) go({ k: w.reason === 'audio_silent' || w.reason === 'audio_too_short' ? 'couldnt' : 'lost' });
            return;
          }
          markWalked();
          let group = '';
          try {
            group = (await getDiary()).entries.find((e) => e.walk_id === walkId)?.group ?? '';
          } catch {
            /* the group name is only decoration */
          }
          if (viewRef.current.k === 'writing' || viewRef.current.k === 'slow') go({ k: 'saved', data: toSaved(w, group) });
          return;
        }
      })();
    },
    [go, stopPolling],
  );

  // ---------------------------------------------------------------- offline queue (FR-J9)
  const drain = useCallback(async () => {
    if (draining.current) return;
    draining.current = true;
    clearTimeout(retryTimer.current);
    try {
      for (const item of await listPending()) {
        try {
          const ack = await postWalk(item.blob);
          await deletePending(item.id); // the hub owns the note now; the phone deletes its copy
          attempt.current = 0;
          if (viewRef.current.k === 'sleeping') startPolling(ack.walk_id);
        } catch (e) {
          if (!isTransient(e)) {
            await deletePending(item.id); // permanent: the hub will never take this recording
            if (viewRef.current.k === 'sleeping') go({ k: 'couldnt' });
            continue;
          }
          const delay = BACKOFF_MS[Math.min(attempt.current++, BACKOFF_MS.length - 1)]!;
          retryTimer.current = setTimeout(() => void drain(), delay);
          return;
        }
      }
    } catch {
      retryTimer.current = setTimeout(() => void drain(), BACKOFF_MS[0]);
    } finally {
      draining.current = false;
    }
  }, [go, startPolling]);

  const submit = useCallback(
    async (blob: Blob, seconds: number) => {
      // NOT IN DESIGN (REVIEW): between release and the hub's 202 the recording screen stays as it was, static and inert.
      setSecs(seconds);
      go({ k: 'sending' });
      try {
        const ack = await postWalk(blob);
        startPolling(ack.walk_id);
      } catch (e) {
        if (!isTransient(e)) return go({ k: 'couldnt' });
        try {
          await addPending(blob, seconds);
        } catch {
          return go({ k: 'unreachable' }); // cannot even queue it: say so rather than pretend
        }
        go({ k: 'sleeping', seconds });
        retryTimer.current = setTimeout(() => void drain(), BACKOFF_MS[0]);
      }
    },
    [drain, go, startPolling],
  );

  useEffect(() => {
    if (!getToken()) return;
    void listPending()
      .then((items) => {
        if (items.length === 0) return;
        if (viewRef.current.k === 'door') go({ k: 'sleeping', seconds: items[0]!.seconds });
        void drain();
      })
      .catch(() => undefined);
    const wake = () => void drain();
    const onVisible = () => {
      if (document.visibilityState === 'visible') wake();
    };
    window.addEventListener('online', wake);
    window.addEventListener('focus', wake);
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      window.removeEventListener('online', wake);
      window.removeEventListener('focus', wake);
      document.removeEventListener('visibilitychange', onVisible);
    };
  }, [drain, go]);

  // ---------------------------------------------------------------- J1: hear the mission twice
  const runPlayer = useCallback(
    async (text: string, url: string | null) => {
      player.current?.stop();
      const n = text.split(' ').length;
      setPlay({ label: 'Playing · 1 of 2', progress: 0, revealed: 0, paused: false });
      go({ k: 'playing' });
      const p = new MissionPlayer(url, n, setPlay);
      player.current = p;
      await p.run();
      if (player.current === p && viewRef.current.k === 'playing') go({ k: 'leave', dimmed: false });
    },
    [go],
  );

  const hear = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    try {
      let m = await getMission();
      for (let i = 0; !m.audio_path && i < 6; i++) {
        await sleep(1500); // TTS renders in the background (FR-M7); the text never waits on it
        m = await getMission();
      }
      setMissionText(m.text);
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
      audioUrl.current = m.audio_path ? await getAudioBlobUrl(m.audio_path) : null;
      await runPlayer(m.text, audioUrl.current);
    } catch {
      go({ k: 'unreachable' });
    } finally {
      busy.current = false;
    }
  }, [go, runPlayer]);

  const tell = useCallback(() => {
    player.current?.stop();
    go({ k: 'tell' });
    if (!missionText) void getMission().then((m) => setMissionText(m.text)).catch(() => undefined);
  }, [go, missionText]);

  const toDoor = useCallback(() => {
    player.current?.stop();
    stopPolling();
    go({ k: 'door' });
  }, [go, stopPolling]);

  // ---------------------------------------------------------------- J3: hold to record
  const finish = useCallback(
    async (p: Press) => {
      clearInterval(p.tick);
      press.current = null;
      const { blob, seconds } = await p.recorder!.stop();
      await submit(blob, seconds);
    },
    [submit],
  );

  const release = useCallback(() => {
    const p = press.current;
    if (!p) return;
    p.released = true;
    if (p.recorder) void finish(p);
  }, [finish]);

  useEffect(() => {
    const up = (e: Event) => {
      if (e instanceof KeyboardEvent && e.key !== ' ' && e.key !== 'Enter') return;
      release();
    };
    window.addEventListener('pointerup', up, true);
    window.addEventListener('pointercancel', up, true);
    window.addEventListener('keyup', up, true);
    return () => {
      window.removeEventListener('pointerup', up, true);
      window.removeEventListener('pointercancel', up, true);
      window.removeEventListener('keyup', up, true);
    };
  }, [release]);

  const pressStart = useCallback(async () => {
    if (press.current) return;
    const p: Press = { released: false };
    press.current = p;
    try {
      p.recorder = await Recorder.start();
    } catch {
      press.current = null;
      return go({ k: 'mic' });
    }
    if (p.released) {
      p.recorder.cancel(); // let go before the microphone was ready: nothing to send
      press.current = null;
      return;
    }
    setSecs(0);
    go({ k: 'recording' });
    p.tick = setInterval(() => {
      const s = p.recorder!.elapsedSeconds();
      setSecs(s);
      if (s >= MAX_SECONDS) release();
    }, 200);
  }, [go, release]);

  // ---------------------------------------------------------------- diary (FR-D1-D4)
  const loadDiary = useCallback(
    async (filterIndex: number, groups: string[]) => {
      try {
        const d = await getDiary(filterIndex > 0 ? groups[filterIndex - 1] : undefined);
        const g = filterIndex === 0 && groups.length === 0 ? [...new Set(d.entries.map((e) => e.group))] : groups;
        setDiary({ data: toDiaryData(d, ['All', ...g], filterIndex), groups: g, filter: filterIndex });
      } catch {
        go({ k: 'unreachable' });
      }
    },
    [go],
  );

  useEffect(() => {
    if (view.k === 'diary' && diary.data === null) void loadDiary(0, []);
  }, [view.k, diary.data, loadDiary]);

  useEffect(() => {
    const onHash = () => {
      if (location.hash.startsWith('#setup')) return go(bootstrap());
      if (!getToken()) return;
      if (location.hash === '#diary') {
        setDiary({ data: null, groups: [], filter: 0 });
        go({ k: 'diary' });
      }
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [go]);

  // ---------------------------------------------------------------- render
  let screen: React.ReactNode;
  switch (view.k) {
    case 'setup':
      // NOT IN DESIGN: no token on this phone yet (set up once through /#setup?token=…).
      screen = <GapState heading="Sair needs to be set up on this phone." body="Open the setup link from whoever looks after Sair." />;
      break;
    case 'door': {
      const today = longDate(new Date());
      screen = hasWalked() ? (
        <DoorFirstWalkOrDoor first={false} date={today} onHear={() => void hear()} onTell={tell} />
      ) : (
        <DoorFirstWalkOrDoor first date={today} onHear={() => void hear()} onTell={tell} />
      );
      break;
    }
    case 'playing': {
      const words = missionText.split(' ').map((w, i) => ({ w, visible: i < play.revealed }));
      screen = (
        <MissionPlaying
          data={{ ...missionPlaying, words, progress: play.progress, playLabel: play.label, pauseLabel: play.paused ? 'Play' /* NOT IN DESIGN */ : missionPlaying.pauseLabel }}
          onPause={() => player.current?.togglePause()}
        />
      );
      break;
    }
    case 'leave':
      screen = (
        <LeavePhone
          data={leavePhone}
          dimmed={view.dimmed}
          onOkay={() => go({ k: 'leave', dimmed: true })}
          onWake={toDoor}
          onHear={() => void runPlayer(missionText, audioUrl.current)}
        />
      );
      break;
    case 'tell':
      screen = <TellSair data={{ ...tellSair, date: longDate(new Date()), mission: missionText }} onPressStart={() => void pressStart()} />;
      break;
    case 'recording': {
      const base = secs >= WARN_AT_SECONDS ? recordingWarning : recording;
      screen = <Recording data={{ ...base, mission: missionText, time: clock(secs), progress: Math.min(1, secs / MAX_SECONDS) }} onPressStart={() => void pressStart()} />;
      break;
    }
    case 'sending': {
      const base = secs >= WARN_AT_SECONDS ? recordingWarning : recording;
      screen = <Recording data={{ ...base, mission: missionText, time: clock(secs), progress: Math.min(1, secs / MAX_SECONDS) }} animate={false} />;
      break;
    }
    case 'writing':
      screen = <Writing data={writing} />;
      break;
    case 'saved':
      // "Read it to me" is skipped: browser speech synthesis may use network voices on some platforms (NFR-P3).
      screen = <Saved data={view.data} showRead={false} onDone={toDoor} />;
      break;
    case 'slow':
      // NOT IN DESIGN (UI-7, PRD J3 step 4): the saved layout without the entry card text, until the entry arrives.
      screen = <Saved data={{ ...savedFixture, date: longDate(new Date()), group: '', entry: 'Your diary entry will be ready soon', facts: [], outcome: undefined }} showRead={false} onDone={toDoor} />;
      break;
    case 'sleeping':
      // NOT IN DESIGN as a built screen: strings are the design's frame 2G (hub-sleeping), laid out with GapState.
      screen = (
        <GapState
          heading="Sair's home computer is sleeping."
          body="Your note will be saved when it wakes up."
          note={['Your note is safe on this phone', `${clock(view.seconds)} · it will send by itself`]}
          primary={{ label: 'Okay', onClick: toDoor }}
        />
      );
      break;
    case 'couldnt':
      // NOT IN DESIGN as a built screen: strings are the design's frame 2F (couldn't hear).
      screen = <GapState heading="I couldn't hear that clearly." body="Sometimes the fan or the traffic is loud. Let's try once more." primary={{ label: 'Try again', onClick: tell }} quiet={{ label: 'Skip for today', onClick: toDoor }} />;
      break;
    case 'lost':
      // NOT IN DESIGN, copy needs owner review (REVIEW M6).
      screen = <GapState heading="Sair couldn't write this one down." body="Your note was not saved. You can tell Sair again." primary={{ label: 'Try again', onClick: tell }} quiet={{ label: 'Skip for today', onClick: toDoor }} />;
      break;
    case 'mic':
      // NOT IN DESIGN, copy needs owner review (REVIEW M5).
      screen = <GapState heading="Sair can't use the microphone." body="Please allow the microphone for this page, then try again." primary={{ label: 'Try again', onClick: tell }} quiet={{ label: 'Skip for today', onClick: toDoor }} />;
      break;
    case 'unreachable':
      // NOT IN DESIGN (REVIEW M3): heading is the design's 2G line; the body line needs owner review.
      screen = <GapState heading="Sair's home computer is sleeping." body="Please try again in a little while." primary={{ label: 'Okay', onClick: toDoor }} />;
      break;
    case 'diary': {
      const d = diary.data;
      screen = d ? (
        <Diary
          data={d}
          onFilter={(i) => void loadDiary(i, diary.groups)}
          empty={d.entries.length === 0 && diary.filter === 0 ? { heading: 'Your first walk will appear here.', body: 'After each walk, Sair writes a few lines for the family.' } : undefined}
        />
      ) : null;
      break;
    }
  }
  return <div style={{ maxWidth: 430, height: '100dvh', margin: '0 auto', position: 'relative' }}>{screen}</div>;
}

function DoorFirstWalkOrDoor({ first, date, onHear, onTell }: { first: boolean; date: string; onHear: () => void; onTell: () => void }) {
  return first ? <DoorFirstWalk data={{ ...doorFirstWalk, date, note: doorFirstWalk.note! }} onHear={onHear} onTell={onTell} /> : <Door data={{ ...door, date }} onHear={onHear} onTell={onTell} />;
}
