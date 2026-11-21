/* Audible activation, a native player, and decoded local clips for timed effects. */
(() => {
  if(document.body.dataset.theme!=='trip'||new URLSearchParams(location.search).has('capture'))return;
  const screams=['scream-female-dry','scream-female-sharp','scream-female-long','scream-male-high','scream-male-deep','scream-crowd-close','scream-crowd-panic','scream-echo','scream-rabbit','scream-beast'];
  const titles={'scream-female-dry':'Крик — высокий','scream-female-sharp':'Крик — резкий','scream-female-long':'Крик — протяжный','scream-male-high':'Крик — мужской','scream-male-deep':'Крик — низкий','scream-crowd-close':'Крики — толпа','scream-crowd-panic':'Крики — паника','scream-echo':'Крик — эхо','scream-rabbit':'Крик — Бонни','scream-beast':'Крик — зверь'};
  let soundBag=[],lastClip=null;
  function nextScream(){
    if(!soundBag.length){
      soundBag=[...screams];
      for(let i=soundBag.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[soundBag[i],soundBag[j]]=[soundBag[j],soundBag[i]];}
      if(soundBag.at(-1)===lastClip)[soundBag[0],soundBag[soundBag.length-1]]=[soundBag[soundBag.length-1],soundBag[0]];
    }
    lastClip=soundBag.pop();return lastClip;
  }
  const audioBase=new URL('../audio/',document.currentScript.src);
  const url=name=>new URL(name+'.wav',audioBase).href;
  const player=document.createElement('audio');player.id='trip-audio-player';
  player.controls=true;player.preload='auto';player.volume=.65;player.src=url(screams[0]);
  let enabled=true,heard=false,gesture=false,request=0,context,gain,source,purpose;
  const buffers=new Map();
  const button=document.createElement('button');button.id='trip-audio-enable';
  const desktop=document.createElement('span');desktop.className='sound-desktop';
  const mobile=document.createElement('span');mobile.className='sound-mobile';mobile.textContent='▶ Звук';
  button.append(desktop,mobile);document.querySelector('.display-tools').prepend(button);
  const test=document.createElement('button');test.id='screamer-sound';test.textContent='▶ Проверить звук';
  const mute=document.createElement('button');mute.id='screamer-mute';
  const label=document.createElement('label');label.className='screamer-volume';
  label.innerHTML='Громкость <input type="range" min="0" max="100" value="65" aria-label="Громкость скримеров">';
  const slider=label.querySelector('input');
  const status=document.createElement('p');status.className='sound-status';status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  const credits=document.createElement('a');credits.className='sound-credits';credits.href='audio/SOURCES.txt';credits.target='_blank';credits.rel='noopener';credits.textContent='Авторы и лицензии криков';
  document.getElementById('scream-now').after(test,mute,label,player,status,credits);
  function update(message){
    desktop.textContent=heard&&enabled?'▶ Проверить звук':'▶ Включить звук';
    button.setAttribute('aria-label','Включить и проверить звук');
    button.title='Всегда проигрывает звуковой пример. Выключатель находится в меню «Показ».';
    mute.textContent=enabled?'Выключить звук':'Включить звук';mute.setAttribute('aria-pressed',String(enabled));
    document.body.dataset.audioState=!enabled?'muted':heard?'ready':'needs-click';
    if(message)status.textContent=message;
  }
  function volume(){return Math.max(0,Math.min(1,Number(slider.value)/100));}
  function startContext(){
    if(!gesture)return null;
    try{
      if(!context||context.state==='closed'){
        const Audio=window.AudioContext||window.webkitAudioContext;
        if(!Audio)return null;
        context=new Audio();gain=context.createGain();gain.gain.value=volume();gain.connect(context.destination);
        buffers.clear();
      }
      // Called synchronously by the user gesture, before fetching or decoding anything.
      if(context.state==='suspended')context.resume().catch(()=>{});
      return context;
    }catch{return null;}
  }
  function activate(event){
    if(event?.isTrusted===false)return;
    gesture=true;if(enabled)startContext();
  }
  function halt(){
    ++request;purpose=null;
    if(source){try{source.stop();}catch{}source.disconnect();source=null;}
    player.pause();try{player.currentTime=0;}catch{}
    document.body.dataset.audioPlaying='false';
  }
  function stop(){if(purpose!=='preview')halt();}
  function playing(name,transport){
    heard=true;document.body.dataset.audioPlaying='true';document.body.dataset.audioClip=name;document.body.dataset.audioTransport=transport;
    update('Играет: '+titles[name]);
  }
  function failed(error){
    document.body.dataset.audioPlaying='false';document.body.dataset.audioError=error?.name||'MediaError';
    update('Нажмите ▶ в аудиоплеере ниже. Если тишина — проверьте звук вкладки и устройства.');
  }
  function decoded(name){
    if(!buffers.has(name)){
      const promise=fetch(url(name)).then(response=>{
        if(!response.ok)throw new Error('Audio file unavailable');return response.arrayBuffer();
      }).then(bytes=>context.decodeAudioData(bytes));
      buffers.set(name,promise);promise.catch(()=>buffers.delete(name));
    }
    return buffers.get(name);
  }
  async function webPlay(name,ticket){
    const ctx=startContext();if(!ctx)return false;
    try{
      const buffer=await decoded(name);
      if(ticket!==request||!enabled||document.hidden||ctx.state!=='running')return false;
      const voice=ctx.createBufferSource();voice.buffer=buffer;gain.gain.value=volume();voice.connect(gain);source=voice;
      voice.onended=()=>{voice.disconnect();if(ticket===request){source=null;purpose=null;document.body.dataset.audioPlaying='false';}};
      voice.start();playing(name,'web-audio');return true;
    }catch{return false;}
  }
  async function nativePlay(name,ticket){
    // No await precedes play(): manual tests preserve the browser's user activation.
    player.muted=false;player.volume=volume();player.src=url(name);player.load();
    try{
      await player.play();
      if(ticket!==request)return false;
      playing(name,'native');return true;
    }catch(error){
      if(ticket!==request||error.name==='AbortError')return false;
      if(await webPlay(name,ticket))return true;
      if(ticket===request)failed(error);return false;
    }
  }
  async function effect(kind,portrait){
    if(!enabled||!gesture||document.hidden)return false;
    halt();purpose='effect';const ticket=request;
    const name=nextScream();
    if(await webPlay(name,ticket))return true;
    if(ticket!==request)return false;
    return nativePlay(name,ticket);
  }
  function preview(){
    gesture=true;enabled=true;
    if(volume()===0)slider.value='65';
    startContext();halt();purpose='preview';
    return nativePlay(nextScream(),request);
  }
  document.addEventListener('pointerdown',activate,{passive:true});document.addEventListener('keydown',activate);
  button.addEventListener('click',preview);test.addEventListener('click',preview);
  mute.addEventListener('click',()=>{if(enabled){enabled=false;halt();update('Звук выключен');}else preview();});
  slider.addEventListener('input',()=>{player.volume=volume();if(gain)gain.gain.value=volume();});
  player.addEventListener('play',()=>{
    // Native controls are an independent, browser-managed way to enable sound.
    gesture=true;enabled=true;if(!purpose)purpose='preview';
    if(source){++request;purpose='preview';try{source.stop();}catch{}source.disconnect();source=null;}
    startContext();
  });
  player.addEventListener('playing',()=>{const name=new URL(player.currentSrc||player.src).pathname.split('/').pop().replace('.wav','');playing(name,'native');});
  player.addEventListener('pause',()=>{if(!source)document.body.dataset.audioPlaying='false';});
  player.addEventListener('ended',()=>{if(!source){purpose=null;document.body.dataset.audioPlaying='false';}});
  player.addEventListener('error',()=>{if(purpose)failed(player.error);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)halt();});window.addEventListener('pagehide',halt);
  window.tripAudio={stop,play:effect};
  update('Нажмите «Включить звук»: сразу прозвучит пример.');
})();
