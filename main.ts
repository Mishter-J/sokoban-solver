// Roomwise push-optimal Sokoban solver.
//
// Strategy: a level is a hallway with independent rooms off it (a box that
// leaves its room can never come back, so each room is its own puzzle and
// the level optimum is the sum of the rooms' optima). Each room is solved
// with A* over pushes -- one macro step per push, never single steps -- with
// an admissible heuristic (minimum-cost assignment of boxes to goals over
// precomputed floor distances), dead-square pruning (reverse reachability
// from the goals) and frozen 2x2-block pruning. A state is the sorted box
// squares plus the lowest square of the player's reachable region, so the
// player's exact square inside its region never multiplies the search.
// Work budgets are counts of expanded states, never the clock, so two runs
// of this program answer byte for byte alike.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (counts of expanded states, deterministic by construction).
const MAX_ROOM_EXPANSIONS = 600_000;
const MAX_LEVEL_EXPANSIONS = 2_500_000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

interface Board {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall (includes the outer frame and padded cells)
  goals: number[]; // sorted
  boxes: number[]; // sorted
  player: number;
}

interface Solution {
  pushes: number;
  moves: string;
}

// ---- parsing ---------------------------------------------------------------

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ---- static analysis -------------------------------------------------------
// Squares from which a box, pulled backwards from a goal over open floor,
// can reach some goal. A box pushed onto any other square is lost.
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DY[k] * b.w + DX[k];
      const y = x - d; // square the box is pulled from
      if (!b.wall[y] && !b.wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Flood fill from `from` over floor cells not occupied by a box; marks the
// visited cells in `seen` with `stamp` and returns the lowest cell index
// reached (the canonical name of the player's region).
function regionLow(
  b: Board,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * b.w + DX[k];
      if (!b.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

// True when the box just pushed to `t` is stuck in a 2x2 block of walls and
// boxes with a box off its goal inside the block.
function frozenBlock(b: Board, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - b.w, t - b.w - 1]) {
    const c = a + 1;
    const d = a + b.w;
    const e = d + 1;
    if (!(b.wall[a] || occupied[a])) continue;
    if (!(b.wall[c] || occupied[c])) continue;
    if (!(b.wall[d] || occupied[d])) continue;
    if (!(b.wall[e] || occupied[e])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[c] && !goal[c]) ||
        (occupied[d] && !goal[d]) || (occupied[e] && !goal[e])) return true;
  }
  return false;
}

// Breadth-first distances from one goal over open floor (boxes ignored).
function distancesFrom(b: Board, goal: number): Int32Array {
  const dist = new Int32Array(b.w * b.h).fill(-1);
  const queue: number[] = [goal];
  dist[goal] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * b.w + DX[k];
      if (!b.wall[q] && dist[q] < 0) {
        dist[q] = dist[p] + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

// ---- A* over pushes --------------------------------------------------------

// Small binary heap ordered by (f, then deeper g first, then insertion order)
// so ties break identically on every run.
class Heap {
  private a: number[][] = [];
  size(): number { return this.a.length; }
  push(e: number[]): void {
    const a = this.a;
    let i = a.length;
    a.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!Heap.less(a[i], a[p])) break;
      const tmp = a[i]; a[i] = a[p]; a[p] = tmp;
      i = p;
    }
  }
  pop(): number[] {
    const a = this.a;
    const top = a[0];
    const last = a.pop();
    if (a.length > 0) {
      a[0] = last;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= a.length) break;
        if (c + 1 < a.length && Heap.less(a[c + 1], a[c])) c++;
        if (!Heap.less(a[c], a[i])) break;
        const tmp = a[i]; a[i] = a[c]; a[c] = tmp;
        i = c;
      }
    }
    return top;
  }
  private static less(x: number[], y: number[]): boolean {
    return x[0] < y[0] || (x[0] === y[0] && (x[1] < y[1] || (x[1] === y[1] && x[2] < y[2])));
  }
}

function solvePushes(b: Board, maxExpansions: number): Solution | null {
  const size = b.w * b.h;
  const goal = new Uint8Array(size);
  for (const g of b.goals) goal[g] = 1;
  const live = liveSquares(b);
  if (b.boxes.some((x) => !live[x])) return null;
  const solved = (bs: number[]): boolean => bs.every((x) => goal[x]);

  // Heuristic: cheapest assignment of boxes to goals over floor distances.
  // Each box needs at least its distance in pushes, and pushes move one box
  // each, so the assignment minimum is an admissible lower bound.
  const distTo = b.goals.map((g) => distancesFrom(b, g));
  const n = b.boxes.length;
  const heuristic = (bs: number[]): number => {
    if (n <= 5) {
      // Exact minimum over all assignments (n! <= 120).
      const perm = b.goals.map((_, i) => i);
      let best = Infinity;
      const rec = (i: number, sum: number): void => {
        if (sum >= best) return;
        if (i === n) { best = sum; return; }
        for (let j = i; j < n; j++) {
          const tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
          const d = distTo[perm[i]][bs[i]];
          if (d >= 0) rec(i + 1, sum + d);
          const tmp2 = perm[i]; perm[i] = perm[j]; perm[j] = tmp2;
        }
      };
      rec(0, 0);
      return best;
    }
    let sum = 0;
    for (const box of bs) {
      let m = Infinity;
      for (const d of distTo) if (d[box] >= 0 && d[box] < m) m = d[box];
      if (m === Infinity) return Infinity;
      sum += m;
    }
    return sum;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const keyOf = (bs: number[], low: number): string => `${low}|${bs.join(',')}`;

  const stBoxes: number[][] = [b.boxes.slice()];
  const stPlayer: number[] = [b.player];
  const stParent: number[] = [-1];
  const stG: number[] = [0];
  const stKey: string[] = [];
  const bestG = new Map<string, number>();
  const idxByKey = new Map<string, number>();
  for (const x of b.boxes) occupied[x] = 1;
  const low0 = regionLow(b, occupied, b.player, seen, ++stamp, stack);
  for (const x of b.boxes) occupied[x] = 0;
  const key0 = keyOf(b.boxes, low0);
  stKey.push(key0);
  bestG.set(key0, 0);
  idxByKey.set(key0, 0);
  if (solved(b.boxes)) return { pushes: 0, moves: '' };

  const heap = new Heap();
  const h0 = heuristic(b.boxes);
  if (h0 === Infinity) return null;
  let seq = 0;
  heap.push([h0, 0, seq++, 0]);

  let found = -1;
  let expansions = 0;
  while (heap.size() > 0 && found < 0) {
    const entry = heap.pop();
    const s = entry[3];
    const g = stG[s];
    if (g !== bestG.get(stKey[s])) continue; // superseded by a cheaper route
    const boxes = stBoxes[s];
    if (solved(boxes)) { found = s; break; } // first popped goal is push-optimal
    if (expansions >= maxExpansions) return null;
    expansions++;
    for (const x of boxes) occupied[x] = 1;
    regionLow(b, occupied, stPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const box = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * b.w + DX[k];
        const t = box + d;
        if (seen[box - d] !== here) continue; // player cannot reach the push square
        if (b.wall[t] || occupied[t] || !live[t]) continue;
        occupied[box] = 0;
        occupied[t] = 1;
        if (!frozenBlock(b, occupied, goal, t)) {
          const next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          const low = regionLow(b, occupied, box, mark, ++stamp, stack);
          const key = keyOf(next, low);
          const prevBest = bestG.get(key);
          if (prevBest === undefined || g + 1 < prevBest) {
            bestG.set(key, g + 1);
            const h = solved(next) ? 0 : heuristic(next);
            if (h !== Infinity) {
              let idx = idxByKey.get(key);
              if (idx === undefined) {
                idx = stBoxes.length;
                idxByKey.set(key, idx);
                stBoxes.push(next);
                stPlayer.push(box);
                stParent.push(s);
                stG.push(g + 1);
                stKey.push(key);
              } else {
                // Cheaper route to a known state: reopen it with the new
                // parent and cost (keeps the search push-optimal even when
                // the heuristic is inadmissible in the inconsistent sense).
                stPlayer[idx] = box;
                stParent[idx] = s;
                stG[idx] = g + 1;
              }
              heap.push([g + 1 + h, -(g + 1), seq++, idx]);
            }
          }
        }
        occupied[t] = 0;
        occupied[box] = 1;
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Reconstruct the moves: walk to each push square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = stParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = b.player;
  let boxes = b.boxes;
  for (const s of path) {
    const nextBoxes = stBoxes[s];
    const from = boxes.find((x) => !nextBoxes.includes(x));
    const to = nextBoxes.find((x) => !boxes.includes(x));
    const d = to - from;
    const k = DIRS_INDEX(d, b.w);
    for (const x of boxes) occupied[x] = 1;
    moves += walkTo(b, occupied, player, from - d);
    for (const x of boxes) occupied[x] = 0;
    moves += PUSH[k];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

function DIRS_INDEX(d: number, w: number): number {
  if (d === -w) return 0;
  if (d === w) return 1;
  if (d === -1) return 2;
  return 3;
}

// Shortest walk from `from` to `to` around walls and boxes, in lower-case
// LURD letters (first direction first on ties, so the walk is deterministic).
function walkTo(b: Board, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.w * b.h).fill(-1);
  const how = new Array<string>(b.w * b.h);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * b.w + DX[k];
      if (b.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP[k];
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

// The square the player stands on after `moves` (case aside) from `from`.
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) {
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  }
  return p;
}

// ---- room decomposition ----------------------------------------------------
// Cut the hallway row's dead squares out of the floor; each remaining part
// holding boxes is a room. Null when the level is not a set of rooms.
function roomsOf(b: Board): { boxes: number[]; goals: number[]; done: boolean }[] | null {
  const size = b.w * b.h;
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(size);
  for (let i = row * b.w; i < (row + 1) * b.w; i++) {
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((x) => cut[x])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: { boxes: number[]; goals: number[]; done: boolean }[] = [];
  for (let s = 0; s < size; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * b.w + DX[k];
        if (!b.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const x of b.boxes) rooms[part[x]].boxes.push(x);
  for (const g of b.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveByRooms(b: Board): Solution | null {
  const rooms = roomsOf(b);
  if (rooms === null) return solvePushes(b, MAX_LEVEL_EXPANSIONS);
  let player = b.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = b.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const x of other.done ? other.goals : other.boxes) wall[x] = 1;
    }
    const sub: Board = {
      w: b.w,
      h: b.h,
      wall,
      goals: room.goals.slice().sort((x, y) => x - y),
      boxes: room.boxes.slice().sort((x, y) => x - y),
      player,
    };
    const r = solvePushes(sub, MAX_ROOM_EXPANSIONS);
    if (r === null) return solvePushes(b, MAX_LEVEL_EXPANSIONS);
    pushes += r.pushes;
    moves += r.moves;
    player = endOf(b.w, player, r.moves);
    room.done = true;
  }
  return { pushes, moves };
}

// ---- verification ----------------------------------------------------------
// Replay `moves` on the level; true only when every step is legal and every
// box ends on a goal. Guards the room-by-room join before an answer ships.
function replaysSolved(b: Board, moves: string): boolean {
  const size = b.w * b.h;
  const goal = new Uint8Array(size);
  for (const g of b.goals) goal[g] = 1;
  const boxes = new Set<number>(b.boxes);
  let player = b.player;
  for (const ch of moves.toLowerCase()) {
    const k = ch === 'u' ? 0 : ch === 'd' ? 1 : ch === 'l' ? 2 : ch === 'r' ? 3 : -1;
    if (k < 0) return false;
    const d = DY[k] * b.w + DX[k];
    const q = player + d;
    if (b.wall[q]) return false;
    if (boxes.has(q)) {
      const t = q + d;
      if (b.wall[t] || boxes.has(t)) return false;
      boxes.delete(q);
      boxes.add(t);
    }
    player = q;
  }
  for (const x of boxes) if (!goal[x]) return false;
  return true;
}

function solveLevel(text: string): string {
  const b = parseLevel(text);
  if (b.player < 0 || b.boxes.length === 0) return '';
  const byRooms = solveByRooms(b);
  if (byRooms !== null && replaysSolved(b, byRooms.moves)) return byRooms.moves;
  const whole = solvePushes(b, MAX_LEVEL_EXPANSIONS);
  if (whole !== null && replaysSolved(b, whole.moves)) return whole.moves;
  return '';
}

// ---- driver ----------------------------------------------------------------

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
