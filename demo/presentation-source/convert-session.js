// Runs in an Artifact Session after the browser has saved site-capture/*.json and PNGs.
let conversion = await artifactSession.run(async ({presentation,session}) => {
 const fs=await import('node:fs/promises');
 const root='/Users/iris/Projects/kokoeto/hackaton-cu-2026/demo/exports/site-capture';
 const cssColor=value=>{const m=value.match(/[\d.]+/g);if(!m)return '#FFFFFF';return '#'+m.slice(0,3).map(n=>Math.round(Number(n)).toString(16).padStart(2,'0')).join('')+(m.length>3?Math.round(Number(m[3])*255).toString(16).padStart(2,'0'):'');};
 const typeface=family=>family.includes('Mono')||family.includes('monospace')?'Menlo':'Helvetica Neue';
 const align=value=>value==='end'?'right':['left','right','center','justify'].includes(value)?value:'left';
 presentation.slides.keep([]);
 presentation.theme.defaultFont='Helvetica Neue';
 for(let i=1;i<=5;i++) {
  const d=JSON.parse(await fs.readFile(root+'/slide-'+i+'.json','utf8'));
  const slide=presentation.slides.add();slide.setViewportSize(1280,720);slide.background.fill='#070C1C';
  slide.images.add({path:root+'/slide-'+i+'-background.png',alt:'Стеклянные карточки и фон из HTML-презентации «Астрамис»',fit:'contain',position:{left:0,top:0,width:1280,height:720}});
  for(const line of d.text) {
   const value=line.transform==='uppercase'?line.value.toUpperCase():line.value;
   const shape=slide.shapes.add({geometry:'textbox',name:'HTML: '+value.trim(),position:{left:line.left,top:line.top,width:line.width+2,height:line.height+2},fill:'none',line:{fill:'none',width:0}});
   shape.text=value;
   shape.text.style={typeface:typeface(line.fontFamily),fontSize:line.fontSize*(line.fontSize>=40||line.letterSpacing<0?.94:.985),bold:line.bold,color:cssColor(line.color),wrap:'none',autoFit:'shrinkText',insets:0,verticalAlignment:'top'};
  }
  for(const source of d.tables) {
   // Each HTML row keeps its own native table, including the exact CSS row gap.
   for(const row of source.rows) {
    const table=slide.tables.add({rows:1,columns:row.cells.length,left:row.left,top:row.top,width:row.width,height:row.height,columnWidths:row.cells.map(c=>c.width),values:[row.cells.map(c=>c.transform==='uppercase'?c.value.toUpperCase():c.value)]});
    table.borders.assign({fill:'none',width:0});
    table.styleOptions={headerRow:false,bandedRows:false,bandedColumns:false};
    table.rows[0].height=row.height;
    row.cells.forEach((cell,c)=>{
     table.cells.block({row:0,column:c,rowCount:1,columnCount:1}).assign({fill:'none',textStyle:{typeface:typeface(cell.fontFamily),fontSize:cell.value==='Бейзлайн'?8:cell.fontSize*.985,bold:cell.bold,color:cssColor(cell.color),alignment:align(cell.alignment)},margins:{left:cell.padding.left,right:cell.fontSize<=12?0:cell.padding.right,top:0,bottom:0},anchor:'center',horizontalOverflow:'overflow'});
     table.getCell(0,c).text.wrap='none';
    });
   }
  }
  slide.speakerNotes.text=d.content+'\n\nИсточник: локальная HTML-презентация demo/index-trip.html. Состав решения пока описывает рабочий план. Значения финальной модели не предоставлены. Бейзлайн 59.6 вычислен из округлённых компонент. Фон и оформление сняты с браузерной вёрстки. Анимации и скримеры доступны в HTML-версии.';
 }
 return {slides:presentation.slides.items.length};
},{artifactType:'presentation',artifactRef:'artifact_U4MS6I38N17ywYX0gr764xfJYFnSD3VM'});
nodeRepl.write(conversion);
