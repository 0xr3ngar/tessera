// Logs the mean and standard deviation of the latents and velocity.
function stats(a) {
  let sum = 0, squares = 0;
  for (let i = 0; i < a.length; i++) { sum += a[i]; squares += a[i] * a[i]; }
  const mean = sum / a.length;
  return `mean ${mean.toFixed(4)} std ${Math.sqrt(squares / a.length - mean * mean).toFixed(4)}`;
}
export function step(s) {
  const head = `${s.phase} ${s.index + 1}/${s.steps} sigma ${s.sigma.toFixed(4)}`;
  if (s.phase == "before") log(head, "latents", stats(s.latents));
  else log(head, "velocity", stats(s.velocity));
}
