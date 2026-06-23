/* =========================================================================
   The Frontier Pretraining Atlas — figure engine + reading chrome (v2)
   Data-driven, themeable SVG figures with motion: scroll-reveal, self-drawing
   charts, tween-on-toggle, plus bespoke animated figures (MRC packet-spray,
   cluster topology, MoE load-imbalance). Scrollspy rail + citation hover-cards.
   No external requests; no math engine. Lineage: MiniMax-M3 Inference Atlas.
   ========================================================================= */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const FAMS = ["compute","comms","moe","mem","prec","data","optim","attn","norm"];
  const motionOK = () => document.documentElement.classList.contains("js-motion") &&
    !(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

  /* ---- color + svg helpers ------------------------------------------ */
  function cssVar(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888"; }
  function famColor(fam){ return cssVar("--c-" + (FAMS.includes(fam) ? fam : "norm")); }
  function famTint(fam){ return cssVar("--t-" + (FAMS.includes(fam) ? fam : "norm")); }
  function ink(){ return cssVar("--ink"); }
  function inkSoft(){ return cssVar("--ink-soft"); }
  function muted(){ return cssVar("--muted"); }
  function line(){ return cssVar("--line"); }
  function paper(){ return cssVar("--paper"); }
  function paper2(){ return cssVar("--paper-2"); }

  function E(tag, attrs, kids){
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs){ if (attrs[k] != null) e.setAttribute(k, attrs[k]); }
    if (kids) (Array.isArray(kids)?kids:[kids]).forEach(k => e.appendChild(typeof k === "string" ? document.createTextNode(k) : k));
    return e;
  }
  function T(x, y, s, o = {}){
    const a = { x, y, "font-size": o.size||13, fill: o.fill||inkSoft(), "text-anchor": o.anchor||"start",
      "font-weight": o.weight||400, "font-family": o.mono ? "var(--mono)" : "var(--sans)", "dominant-baseline": o.base||"alphabetic" };
    if (o.opacity != null) a.opacity = o.opacity;
    return E("text", a, String(s));
  }
  function svgRoot(vw, vh){
    return E("svg", { viewBox: `0 0 ${vw} ${vh}`, width: "100%", preserveAspectRatio: "xMidYMid meet", role: "img" });
  }
  function clear(el){ while (el.firstChild) el.removeChild(el.firstChild); }
  function rr(x,y,w,h,r,fill,o={}){ return E("rect",{x,y,width:Math.max(0,w),height:Math.max(0,h),rx:r,ry:r,fill,
    stroke:o.stroke,"stroke-width":o.sw,opacity:o.opacity,"fill-opacity":o.fillOpacity}); }
  function fmt(n){
    const a = Math.abs(n);
    if (a >= 1e12) return (n/1e12).toFixed(a<1e13?1:0)+"T";
    if (a >= 1e9)  return (n/1e9 ).toFixed(a<1e10?1:0)+"B";
    if (a >= 1e6)  return (n/1e6 ).toFixed(a<1e7 ?1:0)+"M";
    if (a >= 1e3)  return (n/1e3 ).toFixed(a<1e4 ?1:0)+"k";
    if (a > 0 && a < 1) return (+n.toPrecision(2)).toString();
    return (+n.toFixed(1)).toString();
  }
  function arrow(id, color){
    return E("marker",{id,viewBox:"0 0 10 10",refX:8.5,refY:5,markerWidth:7,markerHeight:7,orient:"auto-start-reverse"},
      E("path",{d:"M0,0 L10,5 L0,10 z",fill:color}));
  }
  function legend(frame, items){
    const wrap = document.createElement("div"); wrap.className = "legend";
    items.forEach(it => { const li=document.createElement("span"); li.className="li";
      li.innerHTML = `<span class="sw" style="background:${famColor(it.fam)}"></span>${it.label}`; wrap.appendChild(li); });
    frame.appendChild(wrap);
  }
  function ctrlbar(frame){ const c=document.createElement("div"); c.className="controls"; frame.appendChild(c); return c; }

  /* ---- scales -------------------------------------------------------- */
  function linScale(d0,d1,r0,r1){ const m=(r1-r0)/((d1-d0)||1); return v => r0+(v-d0)*m; }
  function logScale(d0,d1,r0,r1){ const l0=Math.log10(d0),l1=Math.log10(d1),m=(r1-r0)/((l1-l0)||1);
    return v => r0+(Math.log10(v)-l0)*m; }
  function niceTicks(lo,hi,n=5){ const span=hi-lo||1, step0=Math.pow(10,Math.floor(Math.log10(span/n)));
    const err=n*step0/span; let step=step0; if(err<=.15)step*=10; else if(err<=.35)step*=5; else if(err<=.75)step*=2;
    const t=[]; for(let v=Math.ceil(lo/step)*step; v<=hi+1e-9; v+=step) t.push(+v.toFixed(10)); return t; }

  /* ===================================================================
     FIGURE BUILDERS  —  FIGS[type](frame, spec, fig)
     =================================================================== */
  const FIGS = {};

  FIGS.barshare = function(frame, spec){
    const items = spec.items||[]; const total = items.reduce((s,d)=>s+d.value,0)||1;
    const W=1000,H=120, padL=8,padR=8, y=34, bh=44;
    const svg = svgRoot(W,H); let x=padL; const sx=(W-padL-padR)/total;
    svg.appendChild(T(padL, 20, spec.title||"share of total", {size:13, weight:700, fill:inkSoft()}));
    svg.appendChild(T(W-padR, 20, (spec.unit||""), {size:12, anchor:"end", fill:muted()}));
    items.forEach((d)=>{ const w=d.value*sx;
      svg.appendChild(rr(x,y,w-2,bh,4,famColor(d.fam),{}));
      if (w>54) svg.appendChild(T(x+8, y+bh/2-3, d.label.length>16?d.label.slice(0,15)+"…":d.label, {size:11.5, fill:"#fff", weight:600, base:"middle"}));
      if (w>40) svg.appendChild(T(x+8, y+bh/2+12, (100*d.value/total).toFixed(0)+"%", {size:11, fill:"#fff", base:"middle", mono:true, opacity:.9}));
      x+=w; });
    frame.appendChild(svg);
    legend(frame, items.map(d=>({fam:d.fam,label:`${d.label} · ${(100*d.value/total).toFixed(0)}%`})));
    if (spec.note){ const n=document.createElement("div"); n.className="legend"; n.style.color=muted(); n.textContent=spec.note; frame.appendChild(n); }
  };

  FIGS.bars = function(frame, spec){
    const bars = spec.bars||[]; const W=1000, H=420, padL=58, padR=20, padT=22, padB=70;
    const maxV = spec.max || Math.max(...bars.map(b=>b.value), spec.ref?spec.ref.y:0)*1.12;
    const svg = svgRoot(W,H); const y = linScale(0,maxV,H-padB,padT);
    niceTicks(0,maxV,5).forEach(t=>{ svg.appendChild(E("line",{x1:padL,x2:W-padR,y1:y(t),y2:y(t),stroke:line(),"stroke-width":1,opacity:.6}));
      svg.appendChild(T(padL-8,y(t)+4,fmt(t),{size:11,anchor:"end",fill:muted(),mono:true})); });
    if (spec.ylabel){ const ty=padT+(H-padB-padT)/2; const t=T(14,ty,spec.ylabel,{size:12,fill:muted(),anchor:"middle",weight:600}); t.setAttribute("transform",`rotate(-90 14 ${ty})`); svg.appendChild(t); }
    const bw = (W-padL-padR)/bars.length; const innerW = Math.min(bw*0.62, 90);
    bars.forEach((b,i)=>{ const cx=padL+bw*i+bw/2; const h=(H-padB)-y(b.value);
      svg.appendChild(rr(cx-innerW/2,y(b.value),innerW,h,6,famColor(b.fam),{}));
      svg.appendChild(T(cx,y(b.value)-7,fmt(b.value),{size:12,anchor:"middle",weight:700,fill:ink(),mono:true}));
      const lbl=String(b.label).split("\n"); lbl.forEach((ln,j)=> svg.appendChild(T(cx,H-padB+18+j*13,ln,{size:11.5,anchor:"middle",fill:inkSoft()})));
      if (b.sub) svg.appendChild(T(cx,H-padB+18+lbl.length*13,b.sub,{size:10.5,anchor:"middle",fill:muted()}));
    });
    if (spec.ref){ const yr=y(spec.ref.y); svg.appendChild(E("line",{x1:padL,x2:W-padR,y1:yr,y2:yr,stroke:cssVar("--c-comms"),"stroke-width":1.6,"stroke-dasharray":"6 4"}));
      svg.appendChild(T(W-padR,yr-6,spec.ref.label,{size:11,anchor:"end",fill:cssVar("--c-comms"),weight:700})); }
    frame.appendChild(svg);
  };

  FIGS.line = function(frame, spec){
    const S=spec.series||[]; const W=1000,H=440,padL=64,padR=24,padT=24,padB=58;
    const xs=S.flatMap(s=>s.pts.map(p=>p[0])), ys=S.flatMap(s=>s.pts.map(p=>p[1]));
    let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    if(spec.y0!=null)y0=spec.y0; if(spec.logy)y0=Math.min(y0,y1)*0.9; else { const pad=(y1-y0)*.08||1; y0-=pad; y1+=pad; }
    const X=spec.logx?logScale(x0,x1,padL,W-padR):linScale(x0,x1,padL,W-padR);
    const Y=spec.logy?logScale(y0,y1,H-padB,padT):linScale(y0,y1,H-padB,padT);
    const svg=svgRoot(W,H);
    const xt=spec.logx?logTicks(x0,x1):niceTicks(x0,x1,6), yt=spec.logy?logTicks(y0,y1):niceTicks(y0,y1,5);
    yt.forEach(t=>{ svg.appendChild(E("line",{x1:padL,x2:W-padR,y1:Y(t),y2:Y(t),stroke:line(),opacity:.55}));
      svg.appendChild(T(padL-8,Y(t)+4,fmt(t),{size:11,anchor:"end",fill:muted(),mono:true})); });
    xt.forEach(t=>{ svg.appendChild(E("line",{x1:X(t),x2:X(t),y1:padT,y2:H-padB,stroke:line(),opacity:.4}));
      svg.appendChild(T(X(t),H-padB+18,fmt(t),{size:11,anchor:"middle",fill:muted(),mono:true})); });
    S.forEach(s=>{ const d=s.pts.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+" "+Y(p[1]).toFixed(1)).join(" ");
      svg.appendChild(E("path",{d,fill:"none",stroke:famColor(s.fam),"stroke-width":2.6,"stroke-dasharray":s.dash?"7 5":null,"stroke-linejoin":"round","stroke-linecap":"round"}));
      s.pts.forEach(p=> svg.appendChild(E("circle",{cx:X(p[0]),cy:Y(p[1]),r:3,fill:famColor(s.fam),class:"an-fade"}))); });
    (spec.annotations||[]).forEach(a=>{ svg.appendChild(E("circle",{cx:X(a.x),cy:Y(a.y),r:4.5,fill:"none",stroke:ink(),"stroke-width":1.5,class:"an-fade"}));
      svg.appendChild(T(X(a.x)+8,Y(a.y)-6,a.text,{size:11,fill:ink(),weight:600})); });
    if(spec.xlabel) svg.appendChild(T((padL+W-padR)/2,H-12,spec.xlabel,{size:12.5,anchor:"middle",fill:inkSoft(),weight:600}));
    if(spec.ylabel){ const ty=(padT+H-padB)/2; const t=T(16,ty,spec.ylabel,{size:12.5,anchor:"middle",fill:inkSoft(),weight:600}); t.setAttribute("transform",`rotate(-90 16 ${ty})`); svg.appendChild(t);}
    frame.appendChild(svg);
    legend(frame, S.map(s=>({fam:s.fam,label:s.name})));
    function logTicks(a,b){ const t=[]; for(let e=Math.floor(Math.log10(a));e<=Math.ceil(Math.log10(b));e++) t.push(Math.pow(10,e)); return t.filter(v=>v>=a*0.99&&v<=b*1.01); }
  };

  FIGS.roofline = function(frame, spec){
    const W=1000,H=440,padL=70,padR=24,padT=24,padB=58;
    const ridge=spec.peak/spec.bw;
    const pts=spec.points||[]; const aiMin=Math.min(ridge/16,...pts.map(p=>p.ai))*0.6, aiMax=Math.max(ridge*4,...pts.map(p=>p.ai))*1.3;
    const pMin=Math.min(spec.peak/64,...pts.map(p=>p.perf))*0.6, pMax=spec.peak*1.25;
    const X=logScale(aiMin,aiMax,padL,W-padR), Y=logScale(pMin,pMax,H-padB,padT);
    const svg=svgRoot(W,H);
    for(let e=Math.floor(Math.log10(pMin));e<=Math.ceil(Math.log10(pMax));e++){const v=Math.pow(10,e); if(v<pMin||v>pMax)continue;
      svg.appendChild(E("line",{x1:padL,x2:W-padR,y1:Y(v),y2:Y(v),stroke:line(),opacity:.5}));
      svg.appendChild(T(padL-8,Y(v)+4,fmt(v),{size:11,anchor:"end",fill:muted(),mono:true}));}
    const xr=X(ridge), yr=Y(spec.peak);
    svg.appendChild(E("path",{d:`M${padL} ${Y(spec.bw*aiMin)} L${xr} ${yr} L${W-padR} ${yr}`,fill:"none",stroke:cssVar("--c-norm"),"stroke-width":2.4}));
    svg.appendChild(E("line",{x1:xr,x2:xr,y1:padT,y2:H-padB,stroke:cssVar("--c-norm"),"stroke-dasharray":"4 5",opacity:.6}));
    svg.appendChild(T(xr+6,padT+12,`ridge ≈ ${fmt(ridge)} FLOP/byte`,{size:11,fill:muted(),weight:600}));
    svg.appendChild(T(W-padR,yr-7,`peak ${fmt(spec.peak)} FLOP/s`,{size:11,anchor:"end",fill:cssVar("--c-norm"),weight:700}));
    pts.forEach(p=>{ const cx=X(p.ai),cy=Y(p.perf); svg.appendChild(E("circle",{cx,cy,r:6,fill:famColor(p.fam),stroke:paper(),"stroke-width":1.5,class:"an-fade"}));
      svg.appendChild(T(cx+9,cy+4,p.name,{size:11.5,fill:ink(),weight:600})); });
    svg.appendChild(T((padL+W-padR)/2,H-12,"arithmetic intensity — FLOP / byte (log)",{size:12.5,anchor:"middle",fill:inkSoft(),weight:600}));
    frame.appendChild(svg);
  };

  FIGS.membar = function(frame, spec, fig){
    const dtypes=spec.dtypes||[{id:"bf16",w:2,label:"BF16"},{id:"fp8",w:1,label:"FP8"},{id:"fp4",w:0.5,label:"FP4"}];
    const P=(spec.paramsB||200)*1e9; let cur=dtypes[0];
    const cbar=ctrlbar(frame); const lab=document.createElement("label"); lab.innerHTML="weight/grad dtype ";
    const seg=document.createElement("span"); seg.className="seg";
    dtypes.forEach((dt,i)=>{ const b=document.createElement("button"); b.textContent=dt.label; if(i===0)b.className="on";
      b.onclick=()=>{ cur=dt; $$("button",seg).forEach(x=>x.classList.remove("on")); b.classList.add("on"); draw(); };
      seg.appendChild(b); });
    lab.appendChild(seg); cbar.appendChild(lab);
    const host=document.createElement("div"); frame.appendChild(host);
    function partGB(pp){ if(pp==="w")return P*cur.w/1e9; if(pp==="grad")return P*cur.w/1e9;
      if(pp==="opt")return P*(spec.optBytes||8)/1e9; if(pp==="act")return (spec.activGB||0); return P*pp/1e9; }
    function draw(){ clear(host); const parts=(spec.parts||[]).map(p=>({...p,gb:partGB(p.perParam)}));
      const tot=parts.reduce((s,p)=>s+p.gb,0)||1; const W=1000,H=130,padL=8,padR=8,y=40,bh=46;
      const svg=svgRoot(W,H); let x=padL; const sx=(W-padL-padR)/tot;
      svg.appendChild(T(padL,22,`${spec.paramsB||200}B params · ${cur.label} → total state ≈ ${fmt(tot*1e9)}B (${(tot/1000).toFixed(1)} TB)`,{size:13,weight:700,fill:inkSoft()}));
      parts.forEach(p=>{ const w=p.gb*sx; svg.appendChild(rr(x,y,w-2,bh,4,famColor(p.fam),{}));
        if(w>62){svg.appendChild(T(x+8,y+bh/2-3,p.label,{size:11.5,fill:"#fff",weight:600,base:"middle"}));
          svg.appendChild(T(x+8,y+bh/2+12,fmt(p.gb*1e9)+"B",{size:11,fill:"#fff",base:"middle",mono:true,opacity:.9}));} x+=w; });
      host.appendChild(svg);
      legend(host, parts.map(p=>({fam:p.fam,label:`${p.label} · ${(p.gb/1000).toFixed(2)} TB`})));
      reanim(fig, host); }
    draw();
  };

  FIGS.meshlayout = function(frame, spec){
    const axes=spec.axes||[]; const prod=axes.reduce((s,a)=>s*a.size,1);
    const W=1000,H=210; const svg=svgRoot(W,H);
    let x=20; const gap=14; const totW=W-40-gap*(axes.length); const unit=totW/axes.length;
    axes.forEach((a)=>{ const w=unit; svg.appendChild(rr(x,40,w,96,12,famTint(a.fam),{stroke:famColor(a.fam),sw:1.5}));
      svg.appendChild(T(x+w/2,72,a.id,{size:15,anchor:"middle",weight:800,fill:famColor(a.fam)}));
      svg.appendChild(T(x+w/2,96,"×"+a.size,{size:20,anchor:"middle",weight:800,fill:ink(),mono:true}));
      svg.appendChild(T(x+w/2,120,a.name,{size:11,anchor:"middle",fill:muted()}));
      if(x+w<W-40) svg.appendChild(T(x+w+gap/2,90,"×",{size:18,anchor:"middle",fill:muted(),weight:700}));
      x+=w+gap; });
    svg.appendChild(T(W/2,168,`= ${prod.toLocaleString()} devices` + (spec.deviceTotal&&spec.deviceTotal!==prod?`  (target ${spec.deviceTotal.toLocaleString()})`:""),
      {size:16,anchor:"middle",weight:800,fill:ink()}));
    if(spec.note) svg.appendChild(T(W/2,192,spec.note,{size:12,anchor:"middle",fill:muted()}));
    frame.appendChild(svg);
    legend(frame, axes.map(a=>({fam:a.fam,label:`${a.id} — ${a.name}`})));
  };

  FIGS.pipeline = function(frame, spec, fig){
    const stages=spec.stages||8, mb=spec.microbatches||8;
    const scheds=spec.schedules||["1f1b","dualpipe"]; let cur=scheds[0];
    const cbar=ctrlbar(frame); const lab=document.createElement("label"); lab.textContent="schedule ";
    const seg=document.createElement("span"); seg.className="seg";
    scheds.forEach((s,i)=>{ const b=document.createElement("button"); b.textContent=s.toUpperCase(); if(i===0)b.className="on";
      b.onclick=()=>{cur=s;$$("button",seg).forEach(x=>x.classList.remove("on"));b.classList.add("on");draw();}; seg.appendChild(b);});
    lab.appendChild(seg); cbar.appendChild(lab);
    const host=document.createElement("div"); frame.appendChild(host);
    function draw(){ clear(host); const W=1000,padL=58,rowH=30,top=14; const cell=(W-padL-16)/(2*mb+stages);
      const H=top+stages*rowH+34; const svg=svgRoot(W,H);
      let bubble=0,totalCells=0;
      for(let s=0;s<stages;s++){ const y=top+s*rowH;
        svg.appendChild(T(padL-8,y+rowH/2+4,"stage "+(s+1),{size:10.5,anchor:"end",fill:muted(),mono:true}));
        const lead = cur==="dualpipe" ? Math.floor(s/2) : s;
        for(let m=0;m<mb;m++){ const xf=padL+(lead+m)*cell; svg.appendChild(rr(xf,y+3,cell-2,rowH-7,3,famColor("compute"),{fillOpacity:.92}));}
        for(let m=0;m<mb;m++){ const xb=padL+(mb+ (cur==="dualpipe"?Math.floor((stages-1-s)/2):(stages-1-s)) +m)*cell;
          svg.appendChild(rr(xb,y+3,cell-2,rowH-7,3,famColor("comms"),{fillOpacity:.85}));}
        bubble += 2*lead; totalCells += 2*mb;
      }
      svg.appendChild(T(padL, H-10, `${cur.toUpperCase()} · ${stages} stages × ${mb} microbatches · bubble ≈ ${(100*bubble/(totalCells+bubble)).toFixed(0)}% of pipeline`,
        {size:11.5,fill:inkSoft(),weight:600}));
      host.appendChild(svg);
      legend(host,[{fam:"compute",label:"forward"},{fam:"comms",label:"backward"}]);
      reanim(fig, host);
    }
    draw();
  };

  FIGS.flow = function(frame, spec){
    const nodes=spec.nodes||[]; const edges=spec.edges||[];
    const cols=spec.cols||Math.max(...nodes.map(n=>n.col))+1, rows=spec.rows||Math.max(...nodes.map(n=>n.row))+1;
    const W=1000, cw=W/cols, rh=92, padT=18, H=padT+rows*rh+10;
    const svg=svgRoot(W,H); const nw=Math.min(cw*0.74,180), nh=52;
    const defs=E("defs"); FAMS.forEach(f=>defs.appendChild(arrow("ar-"+f,famColor(f)))); svg.appendChild(defs);
    const pos={}; nodes.forEach(n=>{ pos[n.id]={x:n.col*cw+cw/2, y:padT+n.row*rh+nh/2, w:(n.w?nw*n.w:nw), col:n.col}; });
    edges.forEach(e=>{ const a=pos[e.from],b=pos[e.to]; if(!a||!b)return; const fam=e.fam||(nodes.find(n=>n.id===e.from)||{}).fam||"norm";
      const x1=a.x + (b.col>a.col? a.w/2: (b.x>a.x?a.w/2:-a.w/2)), y1=a.y;
      const x2=b.x - (b.x>a.x? b.w/2: -b.w/2), y2=b.y;
      const mx=(x1+x2)/2;
      const dd=`M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`;
      svg.appendChild(E("path",{d:dd,fill:"none",stroke:famColor(fam),"stroke-width":2,
        "marker-end":`url(#ar-${FAMS.includes(fam)?fam:"norm"})`,opacity:.85,"stroke-dasharray":e.dashed?"6 5":null}));
      if(spec.anim){ for(let k=0;k<2;k++){ const dot=E("circle",{r:3.5,fill:famColor(fam),class:"an-fade"});
        dot.appendChild(E("animateMotion",{dur:(2.2+k*0.6).toFixed(1)+"s",begin:(k*0.7).toFixed(1)+"s",repeatCount:"indefinite",path:dd})); svg.appendChild(dot);} }
      if(e.label) svg.appendChild(T(mx,(y1+y2)/2-5,e.label,{size:10.5,anchor:"middle",fill:muted(),mono:true})); });
    nodes.forEach(n=>{ const p=pos[n.id]; svg.appendChild(rr(p.x-p.w/2,padT+n.row*rh,p.w,nh,9,famTint(n.fam),{stroke:famColor(n.fam),sw:1.5}));
      String(n.label).split("\n").forEach((ln,i,arr)=> svg.appendChild(T(p.x,padT+n.row*rh+nh/2 - (arr.length-1)*7 + i*14 +4,ln,
        {size:12,anchor:"middle",weight:600,fill:ink()}))); });
    frame.appendChild(svg);
  };

  FIGS.timeline = function(frame, spec){
    const segs=spec.segments||[]; const tot=segs.reduce((s,d)=>s+d.frac,0)||1;
    const W=1000,H=120,padL=8,padR=8,y=46,bh=40; const svg=svgRoot(W,H); let x=padL; const sx=(W-padL-padR)/tot;
    segs.forEach(d=>{ const w=d.frac*sx; svg.appendChild(rr(x,y,w-2,bh,4,famColor(d.fam),{}));
      svg.appendChild(T(x+w/2,y-7,d.label,{size:11.5,anchor:"middle",weight:700,fill:ink()}));
      if(w>50) svg.appendChild(T(x+w/2,y+bh/2+4,(100*d.frac/tot).toFixed(0)+"%",{size:11,anchor:"middle",fill:"#fff",base:"middle",mono:true}));
      if(d.sub) svg.appendChild(T(x+w/2,y+bh+16,d.sub,{size:10.5,anchor:"middle",fill:muted()})); x+=w; });
    frame.appendChild(svg);
  };

  FIGS.compare = function(frame, spec){
    const host=document.createElement("div"); host.style.display="grid"; host.style.gridTemplateColumns="1fr 1fr"; host.style.gap="14px";
    [spec.left,spec.right].forEach(col=>{ const c=document.createElement("div");
      c.style.cssText=`border:1px solid ${line()};border-top:3px solid ${famColor(col.fam)};border-radius:12px;padding:12px 14px;background:${paper()}`;
      c.innerHTML=`<div style="font-family:var(--sans);font-weight:750;color:${famColor(col.fam)};margin-bottom:6px">${col.title}</div>`+
        "<ul style='margin:0;padding-left:18px;font-size:14px;color:"+inkSoft()+"'>"+(col.items||[]).map(i=>`<li style="margin:4px 0">${i}</li>`).join("")+"</ul>";
      host.appendChild(c); });
    frame.appendChild(host);
    if(spec.rows){ const t=document.createElement("table"); t.className="data"; t.style.marginTop="12px";
      t.innerHTML="<thead><tr><th></th><th>"+spec.left.title+"</th><th>"+spec.right.title+"</th></tr></thead><tbody>"+
        spec.rows.map(r=>`<tr><td><b>${r.k}</b></td><td>${r.left}</td><td>${r.right}</td></tr>`).join("")+"</tbody>";
      const tw=document.createElement("div"); tw.className="tablewrap"; tw.style.margin="12px 0 0"; tw.appendChild(t); frame.appendChild(tw); }
  };

  /* ---- bespoke animated figures ------------------------------------- */
  /* MRC packet-spray: one transfer fanned across many paths, with failover */
  FIGS.packetspray = function(frame, spec){
    const N=spec.paths||8; const W=1000,H=300; const svg=svgRoot(W,H);
    const sx=140, dx=W-140, my=H/2;
    svg.appendChild(rr(40,my-46,100,92,12,famTint("comms"),{stroke:famColor("comms"),sw:1.5}));
    svg.appendChild(T(90,my,"source",{anchor:"middle",weight:700,fill:famColor("comms"),base:"middle"}));
    svg.appendChild(rr(W-140,my-46,100,92,12,famTint("compute"),{stroke:famColor("compute"),sw:1.5}));
    svg.appendChild(T(W-90,my,"dest",{anchor:"middle",weight:700,fill:famColor("compute"),base:"middle"}));
    const failIdx = spec.fail ? Math.floor(N/2) : -1;
    for(let i=0;i<N;i++){ const t=(i+0.5)/N, yy=40+t*(H-90);
      const path=`M${sx} ${my} C ${W*0.4} ${yy}, ${W*0.6} ${yy}, ${dx} ${my}`;
      const failed=i===failIdx;
      svg.appendChild(E("path",{d:path,fill:"none",stroke:failed?famColor("comms"):line(),"stroke-width":failed?2:1.4,"stroke-dasharray":failed?"5 5":null,opacity:failed?.85:.55}));
      if(!failed){ for(let k=0;k<2;k++){ const dot=E("circle",{r:3.4,fill:famColor("data")});
        dot.appendChild(E("animateMotion",{dur:(1.5+i*0.04).toFixed(2)+"s",begin:(k*0.75).toFixed(2)+"s",repeatCount:"indefinite",path:path})); svg.appendChild(dot);} }
      else svg.appendChild(T((sx+dx)/2,yy-7,"× link down — reroute in µs",{anchor:"middle",size:10.5,fill:famColor("comms"),weight:600}));
    }
    svg.appendChild(T(W/2,H-9,`one RDMA transfer · sprayed across ${N} paths`+(failIdx>=0?" · 1 failed, packets reroute":""),{anchor:"middle",size:12,fill:muted(),weight:600}));
    frame.appendChild(svg);
  };

  /* rail-optimized cluster topology: NVL72 domains under leaf/spine */
  FIGS.topology = function(frame, spec){
    const D=spec.domains||6; const W=1000,H=300; const svg=svgRoot(W,H);
    const spineY=44, leafY=148, domY=232; const cw=W/D, spineN=Math.min(4,D);
    for(let i=0;i<spineN;i++){ const x=W*(i+0.5)/spineN; svg.appendChild(rr(x-46,spineY-16,92,32,7,famTint("comms"),{stroke:famColor("comms"),sw:1.3}));
      svg.appendChild(T(x,spineY+1,"spine",{anchor:"middle",size:10.5,fill:famColor("comms"),base:"middle"})); }
    for(let i=0;i<D;i++){ const x=cw*(i+0.5);
      for(let s=0;s<spineN;s++){ const sxp=W*(s+0.5)/spineN; svg.appendChild(E("line",{x1:x,y1:leafY-15,x2:sxp,y2:spineY+16,stroke:line(),"stroke-width":0.8,opacity:.45})); }
      svg.appendChild(E("line",{x1:x,y1:leafY+15,x2:x,y2:domY-15,stroke:line(),"stroke-width":1.3}));
      svg.appendChild(rr(x-46,leafY-15,92,30,7,famTint("data"),{stroke:famColor("data"),sw:1.2})); svg.appendChild(T(x,leafY+1,"leaf",{anchor:"middle",size:10,fill:famColor("data"),base:"middle"}));
      svg.appendChild(rr(x-54,domY-15,108,42,9,famTint("compute"),{stroke:famColor("compute"),sw:1.5})); svg.appendChild(T(x,domY,"NVL72",{anchor:"middle",size:11,weight:700,fill:famColor("compute"),base:"middle"})); svg.appendChild(T(x,domY+15,"72 GPU",{anchor:"middle",size:9.5,fill:muted(),base:"middle"}));
    }
    svg.appendChild(T(W/2,H-8,`rail-optimized fabric · ${D} NVL72 scale-up domains · all-to-all stays inside a domain`,{anchor:"middle",size:12,fill:muted(),weight:600}));
    frame.appendChild(svg);
  };

  /* MoE routing load-imbalance, with aux-loss-free balancing toggle */
  FIGS.loadbalance = function(frame, spec, fig){
    const Nx=spec.experts||12; let bal=false;
    const cbar=ctrlbar(frame); const lab=document.createElement("label"); lab.textContent="balancing ";
    const seg=document.createElement("span"); seg.className="seg";
    ["off","aux-loss-free"].forEach((s,i)=>{ const b=document.createElement("button"); b.textContent=s; if(i===0)b.className="on";
      b.onclick=()=>{bal=i===1;$$("button",seg).forEach(x=>x.classList.remove("on"));b.classList.add("on");draw();}; seg.appendChild(b);});
    lab.appendChild(seg); cbar.appendChild(lab);
    const host=document.createElement("div"); frame.appendChild(host);
    function loads(){ const a=[]; for(let i=0;i<Nx;i++){ a.push(bal ? 0.85+0.3*Math.abs(Math.sin(i*1.7)) : Math.max(0.12, Math.pow(Math.abs(Math.sin(i*2.3+1)),3)*3.2)); }
      const s=a.reduce((x,y)=>x+y,0); return a.map(v=>v/s*Nx); }
    function draw(){ clear(host); const W=1000,H=250,padL=20,padR=20,base=200,top=24; const cw=(W-padL-padR)/Nx; const L=loads(); const cap=1.5;
      const scale=(base-top)/2.4; const svg=svgRoot(W,H);
      svg.appendChild(E("line",{x1:padL,x2:W-padR,y1:base,y2:base,stroke:line()}));
      const capY=base-cap*scale; svg.appendChild(E("line",{x1:padL,x2:W-padR,y1:capY,y2:capY,stroke:famColor("comms"),"stroke-dasharray":"5 4",opacity:.75}));
      svg.appendChild(T(W-padR,capY-5,"capacity",{anchor:"end",size:10,fill:famColor("comms"),weight:600}));
      L.forEach((v,i)=>{ const h=v*scale, over=v>cap; svg.appendChild(rr(padL+cw*i+cw*0.18,base-h,cw*0.64,h,3,over?famColor("comms"):famColor("moe"),{})); });
      svg.appendChild(T(W/2,H-6,bal?"aux-loss-free bias → even load, nothing dropped":"no balancing → hot experts overflow capacity (red); their tokens are dropped",{anchor:"middle",size:12,fill:muted(),weight:600}));
      host.appendChild(svg); reanim(fig, host); }
    draw();
  };

  /* ---- motion plumbing ---------------------------------------------- */
  function tagForMotion(scope){
    if(!motionOK()) return;
    const fig = scope.closest ? scope.closest("figure[data-fig]") : null;
    const revealed = !!(fig && fig.classList.contains("in"));
    $$("svg", scope).forEach(svg=>{
      $$("rect", svg).forEach(r=>{ if(r.dataset.an) return; const w=+r.getAttribute("width")||0, h=+r.getAttribute("height")||0;
        if(w<9||h<9) return; r.dataset.an="1"; r.classList.add(w>=h?"an-grow-h":"an-grow-v"); });
      $$("path,polyline", svg).forEach(p=>{ if(p.dataset.an) return; const f=p.getAttribute("fill"), s=p.getAttribute("stroke");
        if(s && (f==="none"||!f)){ try{ const len=p.getTotalLength(); if(len>5){ p.dataset.an="1"; p.classList.add("an-draw");
          p.setAttribute("stroke-dasharray",len); p.setAttribute("stroke-dashoffset", revealed?0:len); } }catch(e){} } });
    });
  }
  function reanim(fig, scope){
    if(!fig || !motionOK()){ return; }
    tagForMotion(scope || fig);
    fig.classList.remove("in"); void fig.offsetWidth;
    requestAnimationFrame(()=>{ fig.classList.add("in");
      $$(".an-draw", scope||fig).forEach(p=>p.setAttribute("stroke-dashoffset",0)); });
  }

  function readSpec(fig){
    const s=fig.querySelector('script.spec, script[type="application/json"]');
    if(!s) return null; try { return JSON.parse(s.textContent); } catch(e){ console.error("figspec parse", fig.dataset.fig, e); return null; }
  }
  function buildFigure(fig){
    const name=fig.dataset.fig; const spec=readSpec(fig); if(!spec||!FIGS[name]) { if(!FIGS[name]) console.warn("unknown fig type",name); return; }
    let frame=fig.querySelector(".frame");
    if(!frame){ frame=document.createElement("div"); frame.className="frame"; fig.insertBefore(frame, fig.firstChild); }
    clear(frame);
    try { FIGS[name](frame, spec, fig); } catch(e){ console.error("fig", name, e); }
    tagForMotion(frame);
    fig._spec=spec;
  }

  /* ---- syntax highlight (python / js / yaml-ish) -------------------- */
  const KW=new Set("def class return if elif else for while in is not and or with as import from try except finally raise yield lambda global nonlocal assert pass break continue del await async None True False self cls let const var function of new typeof".split(" "));
  const BUILT=new Set("jax jnp np torch nn lax shard_map pjit vmap scan einsum int float bool str list dict tuple range len print super zeros ones array reshape".split(" "));
  function esc(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
  function highlight(code){
    return code.split("\n").map(raw=>{
      let s=raw, comment="";
      const hashIdx=s.indexOf("#"); const slashIdx=s.indexOf("//");
      let ci=-1; if(hashIdx>=0)ci=hashIdx; if(slashIdx>=0&&(ci<0||slashIdx<ci))ci=slashIdx;
      if(ci>=0 && !/["'].*["']/.test(s.slice(0,ci).replace(/[^"']/g,""))){ comment=s.slice(ci); s=s.slice(0,ci); }
      const tokRe=/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|@[A-Za-z_]\w*|\b\d[\d_.eE+-]*\b|\b[A-Za-z_]\w*\b|[^\w\s])/g;
      let m, last=0, out="";
      while((m=tokRe.exec(s))){ out+=esc(s.slice(last,m.index)); const t=m[0]; last=m.index+t.length;
        if(/^["']/.test(t)) out+=`<span class="tok-str">${esc(t)}</span>`;
        else if(/^@/.test(t)) out+=`<span class="tok-dec">${esc(t)}</span>`;
        else if(/^\d/.test(t)) out+=`<span class="tok-num">${esc(t)}</span>`;
        else if(KW.has(t)) out+=`<span class="tok-kw">${esc(t)}</span>`;
        else if(t==="self"||t==="cls") out+=`<span class="tok-self">${esc(t)}</span>`;
        else if(BUILT.has(t)) out+=`<span class="tok-fn">${esc(t)}</span>`;
        else if(/[+\-*/%=<>&|^~]/.test(t)) out+=`<span class="tok-op">${esc(t)}</span>`;
        else out+=esc(t);
      }
      out+=esc(s.slice(last));
      if(comment) out+=`<span class="tok-com">${esc(comment)}</span>`;
      return out;
    }).join("\n");
  }

  /* ---- chrome ------------------------------------------------------- */
  function initTheme(){
    const saved=localStorage.getItem("fp-atlas-theme");
    if(saved) document.documentElement.setAttribute("data-theme",saved);
    const btn=$(".theme-btn");
    if(btn) btn.addEventListener("click",()=>{
      const cur=document.documentElement.getAttribute("data-theme")==="dark"?"":"dark";
      if(cur) document.documentElement.setAttribute("data-theme",cur); else document.documentElement.removeAttribute("data-theme");
      localStorage.setItem("fp-atlas-theme",cur);
      $$("figure[data-fig]").forEach(buildFigure);
    });
  }
  function initProgress(){
    const bar=$(".nav .progress"); if(!bar)return;
    const upd=()=>{ const h=document.documentElement.scrollHeight-window.innerHeight; bar.style.width=(h>0?(window.scrollY/h*100):0)+"%"; };
    window.addEventListener("scroll",upd,{passive:true}); upd();
  }
  function initHighlight(){
    $$("pre.code code, .codecard pre code, pre > code").forEach(c=>{ if(c.dataset.hl)return; c.dataset.hl="1"; c.innerHTML=highlight(c.textContent); });
  }
  function initAnchors(){
    $$("main h2[id], main h3[id]").forEach(h=>{ h.style.cursor="pointer";
      h.addEventListener("click",()=>{ history.replaceState(null,"","#"+h.id); h.scrollIntoView({behavior:"smooth"}); }); });
  }
  function initReveal(){
    const figs=$$("figure[data-fig]");
    const reveals=$$(".callout, .codecard, .stats, .tcards, .formula, .pull, .tablewrap, .sources");
    reveals.forEach(el=>el.classList.add("reveal"));
    const reveal=el=>{ el.classList.add("in"); if(el.matches&&el.matches("figure[data-fig]")) $$(".an-draw",el).forEach(p=>p.setAttribute("stroke-dashoffset",0)); };
    if(!("IntersectionObserver" in window) || !motionOK()){ figs.concat(reveals).forEach(reveal); return; }
    const io=new IntersectionObserver((es)=>{ es.forEach(e=>{ if(e.isIntersecting){ reveal(e.target); io.unobserve(e.target); } }); }, {threshold:0.14, rootMargin:"0px 0px -5% 0px"});
    figs.forEach(f=>io.observe(f)); reveals.forEach(el=>io.observe(el));
    setTimeout(()=>{ figs.concat(reveals).forEach(el=>{ if(!el.classList.contains("in")){ const r=el.getBoundingClientRect(); if(r.top < window.innerHeight*1.15){ reveal(el); io.unobserve(el); } } }); }, 1600);
  }
  function initScrollspy(){
    const hs=$$("main h2[id]"); if(hs.length<3 || window.innerWidth<1400) return;
    const nav=document.createElement("nav"); nav.className="chaptoc";
    nav.innerHTML='<div class="ct-h">On this page</div>';
    const links={};
    hs.forEach(h=>{ const a=document.createElement("a"); a.href="#"+h.id;
      a.textContent=h.textContent.replace(/^[▸\s]*\d+\s*/,"").trim().slice(0,42); links[h.id]=a; nav.appendChild(a); });
    document.body.appendChild(nav);
    if("IntersectionObserver" in window){ const io=new IntersectionObserver((es)=>{ es.forEach(e=>{ if(e.isIntersecting){
      Object.values(links).forEach(x=>x.classList.remove("cur")); if(links[e.target.id]) links[e.target.id].classList.add("cur"); } }); },
      {rootMargin:"-12% 0px -78% 0px"}); hs.forEach(h=>io.observe(h)); }
  }
  function initCites(){
    const card=document.createElement("div"); card.className="citecard"; document.body.appendChild(card); let hideT;
    $$('main a[href^="http"]').forEach(a=>{
      a.addEventListener("mouseenter",()=>{ clearTimeout(hideT); let host; try{host=new URL(a.href).host.replace(/^www\./,"");}catch(e){host=a.href;}
        card.innerHTML='<div class="h">↗ source</div><div style="color:var(--ink);font-weight:600;margin-bottom:3px">'+host+'</div><div class="u">'+a.href.slice(0,90)+'</div>';
        const r=a.getBoundingClientRect(); card.style.left=Math.max(10,Math.min(r.left, window.innerWidth-345))+"px";
        card.style.top=(r.bottom+8 > window.innerHeight-120 ? r.top-8-100 : r.bottom+8)+"px"; card.classList.add("show"); });
      a.addEventListener("mouseleave",()=>{ hideT=setTimeout(()=>card.classList.remove("show"),140); });
    });
  }

  function init(){
    document.documentElement.classList.add("js-motion");
    initTheme();
    $$("figure[data-fig]").forEach(buildFigure);
    initHighlight();
    initReveal();
    initScrollspy();
    initCites();
    initProgress();
    initAnchors();
  }
  if(document.readyState==="loading") document.addEventListener("DOMContentLoaded",init); else init();

  window.FPAtlas = { buildFigure, FIGS, reanim };
})();
