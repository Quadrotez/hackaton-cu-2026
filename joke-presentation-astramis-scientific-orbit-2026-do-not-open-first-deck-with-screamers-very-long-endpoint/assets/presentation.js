(() => {
  const theme = document.body.dataset.theme;
  const capture = new URLSearchParams(location.search).get('capture');
  if (capture) {
    document.body.classList.add('deck-capture');
    document.querySelectorAll('svg').forEach(svg => svg.pauseAnimations?.());
  }
  const header = document.createElement('header');
  header.className = 'site-header';
  header.innerHTML = `<a class="wordmark" href="index.html">ASTRAMIS<span>✳</span></a>
    <nav class="mode-nav" aria-label="Визуальный режим">
      <a href="index-orbit.html" data-mode="orbit">Орбита</a><a href="index-optic.html" data-mode="optic">Оптика</a><a href="index.html" data-mode="trip">Trip</a>
    </nav><div class="display-tools"><button id="motion" aria-pressed="false">Пауза эффектов</button>
    <details class="presenter-menu"><summary>Показ</summary><div class="presenter-popover">
      <button id="overview">Обзор слайдов <kbd>O</kbd></button><button id="laser" aria-pressed="false">Лазерная указка <kbd>L</kbd></button>
      ${theme === 'trip' ? '<button id="faces" aria-pressed="true">Лица включены</button><button id="clear-faces">Сдуть лица <kbd>X</kbd></button><button id="screamers" aria-pressed="true">Скримеры включены</button><button id="scream-now">Скример сейчас</button>' : ''}
      <button id="blackout">Затемнить экран <kbd>B</kbd></button><button id="fullscreen">На весь экран <kbd>F</kbd></button>
      <button id="help">Горячие клавиши <kbd>?</kbd></button>
    </div></details><button id="print">PDF</button></div>`;
  document.body.prepend(header);
  const links = header.querySelectorAll('[data-mode]');
  links.forEach(link => { if (link.dataset.mode === theme) link.setAttribute('aria-current', 'page'); link.addEventListener('click', () => { link.hash = location.hash; }); });
  document.getElementById('prev').addEventListener('click', () => show(current - 1));
  document.getElementById('next').addEventListener('click', () => show(current + 1));
  document.getElementById('print').addEventListener('click', () => window.print());
  document.getElementById('fullscreen').addEventListener('click', async () => {
    try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
    catch { document.getElementById('fullscreen').textContent='Недоступно в браузере'; }
  });
  document.addEventListener('fullscreenchange', () => { document.getElementById('fullscreen').textContent = document.fullscreenElement ? 'Свернуть' : 'На весь экран'; });
  const motion = document.getElementById('motion');
  function setPaused(paused) {
    document.body.classList.toggle('motion-paused', paused);
    motion.setAttribute('aria-pressed', String(paused));
    motion.textContent = paused ? 'Включить эффекты' : 'Пауза эффектов';
    document.querySelectorAll('svg').forEach(svg => { if (paused) svg.pauseAnimations?.(); else svg.unpauseAnimations?.(); });
  }
  motion.addEventListener('click', () => setPaused(!document.body.classList.contains('motion-paused')));
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  setPaused(reduced.matches);
  reduced.addEventListener('change', e => setPaused(e.matches));
  const progress = document.createElement('div');
  progress.className = 'slide-progress';
  progress.setAttribute('aria-hidden', 'true');
  document.body.appendChild(progress);
  const overview = document.createElement('section');
  overview.className = 'deck-overview'; overview.hidden = true;
  overview.setAttribute('role','dialog'); overview.setAttribute('aria-modal','true'); overview.setAttribute('aria-label','Обзор слайдов');
  overview.innerHTML = '<div class="overview-heading"><h2>Вся презентация</h2><button id="close-overview">Закрыть</button></div><div class="overview-list"></div>';
  slides.forEach((slide, i) => {
    const button = document.createElement('button'); button.className = 'overview-item';
    const label = document.createElement('span'); label.textContent = String(i + 1).padStart(2, '0');
    button.append(label, document.createTextNode(slide.querySelector('h1').textContent));
    button.addEventListener('click', () => { show(i); closeOverview(); });
    overview.querySelector('.overview-list').appendChild(button);
  });
  document.body.appendChild(overview);
  const help = document.createElement('section'); help.className='deck-shortcuts';help.hidden=true;
  help.setAttribute('role','dialog');help.setAttribute('aria-modal','true');help.setAttribute('aria-label','Горячие клавиши');
  help.innerHTML='<h2>Управление показом</h2><p>Перелистывать <kbd>← / →</kbd></p><p>Обзор слайдов <kbd>O</kbd></p><p>Указка <kbd>L</kbd></p><p>Затемнить экран <kbd>B</kbd></p><p>На весь экран <kbd>F</kbd></p><p>Сдуть лица <kbd>X</kbd></p><p>Пауза эффектов <kbd>E</kbd></p><p>Печать / PDF <kbd>P</kbd></p><button>Понятно</button>';
  document.body.appendChild(help);
  let lastFocus;
  function closeOverview(){overview.hidden=true;lastFocus?.focus();}
  function openOverview(){lastFocus=document.activeElement;header.querySelector('details').open=false;overview.hidden=false;updateProgress();overview.querySelectorAll('.overview-item')[current].focus();}
  function showHelp(){lastFocus=document.activeElement;header.querySelector('details').open=false;help.hidden=false;help.querySelector('button').focus();}
  function closeHelp(){help.hidden=true;lastFocus?.focus();}
  function updateProgress(){progress.style.width=((current+1)/slides.length*100)+'%';overview.querySelectorAll('.overview-item').forEach((b,i)=>b.setAttribute('aria-current',i===current?'step':'false'));}
  updateProgress();window.addEventListener('slide-change',()=>{
    updateProgress();
    if(matchMedia('(max-width:760px) and (orientation:portrait)').matches) window.scrollTo({top:0,behavior:'instant'});
  });
  document.getElementById('overview').addEventListener('click',openOverview);
  document.getElementById('close-overview').addEventListener('click',closeOverview);
  document.getElementById('help').addEventListener('click',showHelp);help.querySelector('button').addEventListener('click',closeHelp);
  const laser=document.createElement('div');laser.className='laser-pointer';laser.setAttribute('aria-hidden','true');document.body.appendChild(laser);
  document.getElementById('laser').addEventListener('click',()=>{const on=document.body.classList.toggle('laser-enabled');document.getElementById('laser').setAttribute('aria-pressed',String(on));header.querySelector('details').open=false;});
  window.addEventListener('pointermove',e=>{laser.style.left=e.clientX+'px';laser.style.top=e.clientY+'px';laser.classList.add('is-visible');});
  window.addEventListener('pointerout',e=>{if(!e.relatedTarget)laser.classList.remove('is-visible');});
  document.getElementById('blackout').addEventListener('click',()=>{document.body.classList.toggle('deck-blackout');header.querySelector('details').open=false;});
  document.getElementById('clear-faces')?.addEventListener('click',()=>{window.dispatchEvent(new Event('clear-faces'));header.querySelector('details').open=false;});
  document.getElementById('faces')?.addEventListener('click',()=>{const hidden=document.body.classList.toggle('faces-hidden');document.getElementById('faces').textContent=hidden?'Лица выключены':'Лица включены';document.getElementById('faces').setAttribute('aria-pressed',String(!hidden));});
  document.getElementById('screamers')?.addEventListener('click',()=>{
    const off=document.body.classList.toggle('screamers-off');
    document.getElementById('screamers').textContent=off?'Скримеры выключены':'Скримеры включены';
    document.getElementById('screamers').setAttribute('aria-pressed',String(!off));
    document.getElementById('screamer')?.classList.remove('on');
  });
  document.getElementById('scream-now')?.addEventListener('click',()=>{
    header.querySelector('details').open=false;
    window.dispatchEvent(new Event('scream-now'));
  });
  window.addEventListener('keydown',e=>{
    if(e.target.closest('input,textarea,select,[contenteditable]')||e.metaKey||e.ctrlKey||e.altKey)return;
    if(e.key==='Escape'){document.body.classList.remove('deck-blackout');if(!overview.hidden)closeOverview();if(!help.hidden)closeHelp();header.querySelector('details').open=false;return;}
    if(e.key==='Tab'&&(!overview.hidden||!help.hidden)){const dialog=!overview.hidden?overview:help;const buttons=[...dialog.querySelectorAll('button')];if(e.shiftKey&&document.activeElement===buttons[0]){e.preventDefault();buttons.at(-1).focus();}else if(!e.shiftKey&&document.activeElement===buttons.at(-1)){e.preventDefault();buttons[0].focus();}return;}
    if(e.key==='?'){e.preventDefault();if(help.hidden)showHelp();else closeHelp();return;}
    if(!help.hidden||!overview.hidden){if(e.key.toLowerCase()==='o'&&!overview.hidden)closeOverview();return;}
    const ids={o:'overview',l:'laser',b:'blackout',f:'fullscreen',e:'motion',x:'clear-faces'};
    const id=ids[e.key.toLowerCase()];if(id){e.preventDefault();document.getElementById(id)?.click();}
  });
  window.addEventListener('hashchange', () => { const match = location.hash.match(/^#slide-([1-5])$/); if(match) show(Number(match[1])-1); });
  let touchStart;
  stage.addEventListener('touchstart', e => { if (!e.target.closest('a,button')) touchStart = {x:e.touches[0].clientX,y:e.touches[0].clientY}; }, {passive:true});
  stage.addEventListener('touchend', e => { if(!touchStart) return; const dx=e.changedTouches[0].clientX-touchStart.x,dy=e.changedTouches[0].clientY-touchStart.y; if(Math.abs(dx)>70 && Math.abs(dx)>Math.abs(dy)*1.5) show(current+(dx<0?1:-1)); touchStart=null; }, {passive:true});
})();
