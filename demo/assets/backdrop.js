/* Bake decorative SVG geometry once; animate cached image layers afterwards.
   Capture mode keeps the original vectors for the presentation exporter. */
(() => {
  if (new URLSearchParams(location.search).has('capture')) return;
  const theme = document.body.dataset.theme;
  const backdrop = document.getElementById('backdrop');
  const sources = [...backdrop.querySelectorAll(theme === 'trip'
    ? '#trip-a, #trip-b, #trip-c' : theme === 'optic' ? '#spokes, #spiral' : '.orbit-spin')];
  const size = Math.min(2048, Math.max(1024, Math.ceil(Math.max(innerWidth, innerHeight) * Math.min(devicePixelRatio, 1.25))));
  const ns = 'http://www.w3.org/2000/svg';
  async function bake(source) {
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('xmlns', ns);
    svg.setAttribute('viewBox', '-280 -550 2000 2000');
    svg.setAttribute('width', size); svg.setAttribute('height', size);
    const shape = source.cloneNode(true);
    shape.removeAttribute('id'); shape.removeAttribute('class');
    if (theme === 'trip') {
      const defs = document.createElementNS(ns, 'defs');
      const warp = document.getElementById('trip-warp').cloneNode(true);
      warp.querySelectorAll('animate').forEach(node => node.remove());
      defs.appendChild(warp); svg.appendChild(defs);
      shape.setAttribute('filter', 'url(#trip-warp)');
    }
    svg.appendChild(shape);
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type:'image/svg+xml'}));
    try {
      const image = new Image(); image.src = url; await image.decode();
      const canvas = document.createElement('canvas'); canvas.width = canvas.height = size;
      canvas.getContext('2d').drawImage(image, 0, 0, size, size);
      const layer = new Image();
      layer.alt = ''; layer.className = 'backdrop-layer ' + source.getAttribute('class');
      layer.src = canvas.toDataURL('image/png'); await layer.decode();
      return layer;
    } finally { URL.revokeObjectURL(url); }
  }
  Promise.all(sources.map(bake)).then(layers => {
    const host = document.createElement('div');
    host.className = 'backdrop-motion' + (theme === 'trip' ? ' trip-hue' : '');
    host.setAttribute('aria-hidden', 'true');
    host.append(...layers); backdrop.prepend(host);
    // Keep the static decorations and optical guide lines in their original SVG.
    sources.forEach(source => { source.style.display = 'none'; });
    if (theme === 'trip') {
      backdrop.querySelector(':scope > svg').style.display = 'none';
      document.querySelectorAll('#trip-warp animate').forEach(node => node.remove());
    } else {
      backdrop.querySelector(':scope > svg > rect')?.setAttribute('fill-opacity', '0');
    }
    document.body.classList.add('backdrop-cached');
  }).catch(() => { /* Original SVG remains intact when image decoding is unavailable. */ });
})();
