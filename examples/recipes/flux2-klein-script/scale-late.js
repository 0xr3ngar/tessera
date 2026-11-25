// Scales the velocity by 0.8 in the second half of the steps.
export function step(s) {
  if (s.phase != "after" || s.index < s.steps / 2) return;
  for (let i = 0; i < s.velocity.length; i++) s.velocity[i] *= 0.8;
}
