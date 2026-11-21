/* Offline screamer director. Faces use photographic expression frames; sounds are local WAV files. */
(() => {
  if (document.body.dataset.theme!=='trip' || new URLSearchParams(location.search).has('capture')) return;
  const {actor,pick}=window.tripFaces;
  const overlay=document.createElement('div');overlay.id='screamer';overlay.setAttribute('aria-hidden','true');
  const aura=document.createElement('div');aura.className='scream-aura';overlay.appendChild(aura);
  const orbit=document.createElement('div');orbit.className='scream-orbit';
  const flip=document.createElement('div');flip.className='scream-flip';
  const deform=document.createElement('div');deform.className='scream-deform';
  flip.appendChild(deform);orbit.appendChild(flip);overlay.appendChild(orbit);document.body.appendChild(overlay);
  const variants=['impact','spin','flip','pancake','stretch','rgb','haunted'];
  let previousVariant=-1,previousPortrait=null,endTimer,nextTimer,runId=0;
  const stopSound=()=>window.tripAudio?.stop();
  const playSound=(kind,portrait)=>window.tripAudio?.play(kind,portrait);
  function choose(length,last){return last<0?Math.floor(Math.random()*length):(last+1+Math.floor(Math.random()*(length-1)))%length;}
  function stop(force=false) {
    if(!force&&overlay.dataset.locked==='true')return;
    ++runId;clearTimeout(endTimer);overlay.classList.remove('on');document.body.classList.remove('screaming');stopSound();
  }
  function blocked(){return document.hidden||document.body.matches('.motion-paused,.screamers-off,.deck-capture,.deck-blackout')||document.querySelector('.deck-overview:not([hidden]),.deck-shortcuts:not([hidden]),.presenter-menu[open]');}
  function scream() {
    if(blocked()||overlay.classList.contains('on'))return;
    const ticket=++runId;
    previousVariant=choose(variants.length,previousVariant);
    const kind=variants[previousVariant],portrait=pick(previousPortrait);
    previousPortrait=portrait.id;
    // Decode before entering the short animation, so a cold image cannot appear blank.
    const image=new Image();image.src=portrait.src;
    image.decode().then(()=>{
      if(ticket!==runId||blocked()||overlay.classList.contains('on'))return;
      const locked=kind==='haunted'||Math.random()<.2;
      const duration=kind==='haunted'?1900:kind==='spin'?1600:1250+Math.floor(Math.random()*350);
      overlay.dataset.variant=kind;overlay.dataset.locked=String(locked);overlay.dataset.portrait=portrait.id;
      overlay.style.setProperty('--scream-duration',duration+'ms');overlay.style.setProperty('--direction',Math.random()<.5?'-1':'1');
      deform.replaceChildren(actor(portrait,true));overlay.classList.add('on');document.body.classList.add('screaming');
      playSound(kind,portrait.id);endTimer=setTimeout(()=>stop(true),duration);
    }).catch(()=>{});
  }
  function schedule(){clearTimeout(nextTimer);nextTimer=setTimeout(()=>{scream();schedule();},3000+Math.random()*2000);}
  overlay.addEventListener('click',()=>stop());
  window.addEventListener('scream-now',scream);
  window.addEventListener('keydown',e=>{if(e.key==='Escape')stop();});
  document.getElementById('screamers').addEventListener('click',()=>stop(true));
  window.addEventListener('effects-change',()=>{if(document.body.matches('.motion-paused,.screamers-off'))stop(true);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stop(true);clearTimeout(nextTimer);}else schedule();});
  // Covers overview/blackout opened by keyboard while a screamer is playing.
  new MutationObserver(()=>{if(overlay.classList.contains('on')&&blocked())stop(true);}).observe(document.body,{attributes:true,attributeFilter:['class','hidden','open'],subtree:true});
  window.addEventListener('pagehide',()=>{stop(true);clearTimeout(nextTimer);});
  window.addEventListener('pageshow',event=>{if(event.persisted)schedule();});
  schedule();
})();
