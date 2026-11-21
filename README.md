# Roomwise Push-Optimal Sokoban Solver

A Sokoban solver that answers a whole set of levels in one run, in as few
pushes as possible.

## How it works

A level is a hallway with independent rooms off it: a box pushed out of its
room can never be pushed back, so each room is its own puzzle and the
level's push-optimal count is the sum of the rooms' optima. The solver
therefore splits each level into rooms (cutting the hallway row's dead
squares out of the floor) and solves each room on its own, with the other
rooms' boxes standing as walls; the player simply walks on from where the
last room left it.

Each room is solved with A* over **pushes** (one macro step per push, never
single steps):

- **Heuristic:** the minimum-cost assignment of boxes to goals over
  precomputed floor distances (exact Hungarian minimum for small rooms).
  Each box needs at least its distance in pushes, and each push moves one
  box, so this is an admissible lower bound — the search is push-optimal.
- **Dead squares:** squares from which no reverse pull from a goal can
  reach any goal; a box pushed onto one is pruned.
- **Frozen 2x2 blocks:** a block of walls and boxes holding a box off its
  goal is pruned.
- **State compression:** a state is the sorted box squares plus the lowest
  square of the player's reachable region, so the player's exact square
  inside its region never multiplies the search.
- **Work budgets, never the clock:** search is cut off by counts of
  expanded states, so two runs answer byte for byte alike.

Every answer is replayed internally before it is written; if the
room-by-room answer ever failed the replay, the solver falls back to a
whole-level search.

## Run

```sh
node main.ts
```

It reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt` (player moves in LURD, upper-case letters mark
pushes). For local testing, `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR`
override the paths. Node 22, no dependencies, no network.

## Practice result

On the kit's practice set the solver scores **1,000,000 / 1,000,000**:
every instance solved in its push-optimal count (13, 16, 29, 47 and 60
pushes), against 200,000 for the house plain-A* baseline. The whole set
runs in about a second.

## License

MIT — see `LICENSE`.
