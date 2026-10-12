# My Personal Notes for Debugging

## Line Search

Orsolya's line search seems to work even for my points of failure. But it seems to be a "silent failure" since it actually computes the line search alpha to be 0. If the search direction is not a descent direction, it immediately goes to the `zoom()` function on the interval `[0, 1]` and bisects to the left until the interval is so small that it returns `0`. This naturally causes the next step of Newton's method to stop due to the small step taken.

Lukas' Newton's Method works for two of them, but still crashes on the third:
```
x0 = [-1.18315709,  1.40683587]
```