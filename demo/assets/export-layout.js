/* Read the rendered HTML into presentation coordinates. No app data is uploaded. */
(async () => {
  const capture = new URLSearchParams(location.search).get('capture');
  if (!capture) return;
  await document.fonts.ready;
  await Promise.all([...document.images].map(image => image.decode().catch(() => {})));
  await new Promise(resolve => requestAnimationFrame(resolve));
  document.querySelectorAll('svg').forEach(svg => { svg.pauseAnimations?.(); try { svg.setCurrentTime?.(3); } catch {} });
  const slide = document.querySelector('.slide.active');
  const rect = element => {
    const r = element.getBoundingClientRect();
    return {left:r.left,top:r.top,width:r.width,height:r.height};
  };
  function style(element) {
    const c = getComputedStyle(element);
    return {fontSize:parseFloat(c.fontSize),fontFamily:c.fontFamily,bold:Number(c.fontWeight)>=600,color:c.color,
      letterSpacing:parseFloat(c.letterSpacing)||0,lineHeight:parseFloat(c.lineHeight)||parseFloat(c.fontSize)*1.2,
      alignment:c.textAlign,transform:c.textTransform,padding:{left:parseFloat(c.paddingLeft),right:parseFloat(c.paddingRight),top:parseFloat(c.paddingTop),bottom:parseFloat(c.paddingBottom)}};
  }
  const text = [];
  const walker = document.createTreeWalker(slide,NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const parent = node.parentElement;
    if(parent.closest('table,script,style') || !parent.getClientRects().length) continue;
    const cs = style(parent);
    const range = document.createRange();
    let segment;
    for(let i=0;i<node.textContent.length;i++) {
      range.setStart(node,i);range.setEnd(node,i+1);
      const r=range.getBoundingClientRect();
      const char=node.textContent[i].replace(/\s/,' ');
      if(!r.width || !r.height) continue;
      // Store tracked headings and labels word by word, body copy line by line.
      const split=cs.letterSpacing!==0;
      if(!segment || Math.abs(segment.top-r.top)>2 || split) {
        segment={value:'',left:r.left,top:r.top,width:0,height:r.height,...cs};
        text.push(segment);
      }
      segment.value+=char;
      segment.width=Math.max(segment.width,r.right-segment.left);
    }
  }
  // List markers belong to the native text layer too.
  slide.querySelectorAll('li').forEach(li=>{const r=rect(li),cs=style(li);text.push({value:'•',left:r.left-16,top:r.top,width:14,height:cs.lineHeight,...cs});});
  const tables=[...slide.querySelectorAll('table')].map(table=>({
    ...rect(table),rows:[...table.rows].map(row=>({...rect(row),cells:[...row.cells].map(cell=>({...rect(cell),value:cell.textContent.replace(/\s+/g,' ').trim(),...style(cell)}))}))
  }));
  const result={title:slide.querySelector('h1').textContent,slide:Number(location.hash.match(/slide-(\d+)/)?.[1]||1),width:1280,height:720,text:text.filter(t=>t.value.trim()),tables,content:slide.innerText};
  const data=document.createElement('script');data.type='application/json';data.id='slide-export-layout';data.textContent=JSON.stringify(result);document.body.appendChild(data);
  if(capture==='background') document.body.classList.add('capture-background');
  document.body.dataset.exportReady='true';
})();
