/** Zoom 웹/웹클라이언트 호스트 (us05web.zoom.us 등) */
export function isZoomHost(hostname = location.hostname) {
  return (
    /\.zoom\.(us|com)$/i.test(hostname) ||
    /\.zoomgov\.(us|com)$/i.test(hostname)
  );
}
