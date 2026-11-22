/* Photographic expression atlases. Each square image contains four aligned poses:
   closed mouth / open mouth / blink / open mouth + blink. */
(() => {
  const portraits = [
    {id:'face1',src:'faces/expressions/face1-fuller.png',frames:true},
    {id:'face2',src:'faces/face2.png',mouth:71,fuller:true},
    {id:'face3',src:'faces/expressions/face3-fuller.png',frames:true},
    {id:'freddy',src:'faces/freddy.png',mouth:74},
    {id:'racer',src:'faces/expressions/racer.png',frames:true},
    {id:'giraffe',src:'faces/expressions/giraffe.png',frames:true,weight:2},
    {id:'bonnie',src:'faces/expressions/bonnie.png',frames:true,weight:2},
    {id:'fox',src:'faces/expressions/fox.png',frames:true,weight:2},
    {id:'escapee-orange',src:'faces/expressions/escapee-orange.png',frames:true,weight:2},
    {id:'escapee-stripes',src:'faces/expressions/escapee-stripes.png',frames:true,weight:2}
  ];
  function pick(previousId) {
    const choices=portraits.filter(p=>p.id!==previousId);
    let value=Math.random()*choices.reduce((sum,p)=>sum+(p.weight||1),0);
    for(const portrait of choices){value-=portrait.weight||1;if(value<0)return portrait;}
    return choices[choices.length-1];
  }
  function actor(portrait,intense=false) {
    const face=document.createElement('div');
    face.className='face-actor'+(portrait.frames?' face-frames':' face-mechanical')+(portrait.fuller?' face-fuller-cutout':'')+(intense?' face-intense':'');
    face.dataset.portrait=portrait.id;
    if(portrait.fuller)face.style.setProperty('--portrait-position','center 90%');
    if(portrait.frames){
      const frame=document.createElement('div');frame.className='portrait-frame';
      frame.style.backgroundImage=`url("${portrait.src}")`;
      frame.style.animationDelay=intense?'0s':(-Math.random()*2.8)+'s';
      face.appendChild(frame);
    }else{
      // The animatronic's lower jaw is an actual piece of the original image.
      face.style.setProperty('--mouth-y',portrait.mouth+'%');
      for(const part of ['upper','jaw']){
        const image=new Image();image.src=portrait.src;image.alt='';image.draggable=false;
        image.className='face-'+part;face.appendChild(image);
      }
    }
    return face;
  }
  portraits.forEach(p=>{const image=new Image();image.src=p.src;image.decode().catch(()=>{});});
  window.tripFaces={portraits,actor,pick};
})();
