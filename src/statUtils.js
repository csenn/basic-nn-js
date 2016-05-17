// http://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
export function gaussian(mean, stdev) {
  let y2;
  let useLast = false;
  return function () {
    let y1;
    if (useLast) {
      y1 = y2;
      useLast = false;
    }
    else {
      let x1, x2, w;
      do {
        x1 = 2.0 * Math.random() - 1.0;
        x2 = 2.0 * Math.random() - 1.0;
        w = x1 * x1 + x2 * x2;
      } while (w >= 1.0);
      w = Math.sqrt((-2.0 * Math.log(w)) / w);
      y1 = x1 * w;
      y2 = x2 * w;
      useLast = true;
    }

    const retval = mean + stdev * y1;
    return retval;
  };
}
