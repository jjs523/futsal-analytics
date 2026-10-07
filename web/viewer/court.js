// FIFA futsal pitch markings (mirror of src/futsal/court.py). x along the touchline, y along the goal line, metres.
const Court = (() => {
  const CENTRE_R = 3, PA_R = 6, POST_OUT = 1.58, PEN1 = 6, PEN2 = 10;
  const GOAL_W = 3, GOAL_D = 1, SUB_FROM = 5, SUB_LEN = 5, SUB_MARK = 0.8, CORNER_R = 0.25;

  function arc(cx, cy, r, a0, a1, n = 40) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = (a0 + (a1 - a0) * i / (n - 1)) * Math.PI / 180;
      out.push([cx + r * Math.cos(t), cy + r * Math.sin(t)]);
    }
    return out;
  }

  function markings(L, W) {
    const cy = W / 2;
    const lines = [[[0, 0], [L, 0], [L, W], [0, W], [0, 0]], [[L / 2, 0], [L / 2, W]], arc(L / 2, cy, CENTRE_R, 0, 360, 80)];
    for (const [gx, s] of [[0, 1], [L, -1]]) {
      lines.push(s > 0 ? arc(gx, cy + POST_OUT, PA_R, 90, 0) : arc(gx, cy + POST_OUT, PA_R, 90, 180));
      lines.push([[gx + s * PA_R, cy + POST_OUT], [gx + s * PA_R, cy - POST_OUT]]);
      lines.push(s > 0 ? arc(gx, cy - POST_OUT, PA_R, 0, -90) : arc(gx, cy - POST_OUT, PA_R, 180, 270));
    }
    for (const [[cx, cyy], a0] of [[[0, 0], 0], [[L, 0], 90], [[L, W], 180], [[0, W], 270]]) lines.push(arc(cx, cyy, CORNER_R, a0, a0 + 90, 10));
    const h = L / 2;
    for (const x of [h - SUB_FROM - SUB_LEN, h - SUB_FROM, h + SUB_FROM, h + SUB_FROM + SUB_LEN]) lines.push([[x, -SUB_MARK / 2], [x, SUB_MARK / 2]]);
    return lines;
  }

  function spots(L, W) {
    const cy = W / 2;
    return [[L / 2, cy], [PEN1, cy], [PEN2, cy], [L - PEN1, cy], [L - PEN2, cy]];
  }

  function goals(L, W) {
    const y0 = W / 2 - GOAL_W / 2;
    return [{ x: -GOAL_D, y: y0, w: GOAL_D, h: GOAL_W, line: 0 }, { x: L, y: y0, w: GOAL_D, h: GOAL_W, line: L }];
  }

  return { markings, spots, goals, GOAL_W };
})();
